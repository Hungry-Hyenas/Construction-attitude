import re
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd

# KoNLPy: Kkma for sentence splitting
# !pip install konlpy  # Make sure Konlpy is installed, or comment this line if already installed
from konlpy.tag import Kkma

# Mamba 관련 라이브러리
from mamba_ssm import Mamba

# (Optional) For embeddings if still needed
#from langchain.embeddings import HuggingFaceEmbeddings

# For a LangChain wrapper
from langchain.llms.base import LLM
from typing import Optional, List

#####################################################
# 1. 데이터 로드 및 전처리 (train.csv, test.csv)
#####################################################
train = pd.read_csv("./open/train.csv", encoding="utf-8-sig")
test = pd.read_csv("./open/test.csv", encoding="utf-8-sig")

# '공사종류' 전처리
train["공사종류(대분류)"] = train["공사종류"].str.split(" / ").str[0]
train["공사종류(중분류)"] = train["공사종류"].str.split(" / ").str[1]
test["공사종류(대분류)"] = test["공사종류"].str.split(" / ").str[0]
test["공사종류(중분류)"] = test["공사종류"].str.split(" / ").str[1]

# '공종' 전처리
train["공종(대분류)"] = train["공종"].str.split(" > ").str[0]
train["공종(중분류)"] = train["공종"].str.split(" > ").str[1]
test["공종(대분류)"] = test["공종"].str.split(" > ").str[0]
test["공종(중분류)"] = test["공종"].str.split(" > ").str[1]

# '사고객체' 전처리
train["사고객체(대분류)"] = train["사고객체"].str.split(" > ").str[0]
train["사고객체(중분류)"] = train["사고객체"].str.split(" > ").str[1]
test["사고객체(대분류)"] = test["사고객체"].str.split(" > ").str[0]
test["사고객체(중분류)"] = test["사고객체"].str.split(" > ").str[1]

# (Q, A) 튜플 리스트 생성
train_data = []
for _, row in train.iterrows():
    q = (
        f"공종 중분류 '{row['공종(중분류)']}'에서 "
        f"작업 프로세스 '{row['작업프로세스']}' 와 관련된 사고가 발생했습니다. "
        f"사고 원인은 '{row['사고원인']}'입니다. 재발 방지 대책 및 향후 조치 계획은 무엇인가요?"
    )
    a = row["재발방지대책 및 향후조치계획"]
    train_data.append((q, a))

#####################################################
# 2. Kkma 기반 커스텀 "SentenceTokenizer"
#####################################################
class KkmaSentenceTokenizer:
    """
    Splits text into sentence-level "tokens" using Kkma.
    Then maps these sentences to integer IDs for model training.
    (For real-world usage, consider morphological tokens (kkma.morphs)
     rather than entire sentences.)
    """
    def __init__(self, special_tokens=None):
        if special_tokens is None:
            # Define special tokens for [PAD], [SEP], [EOS], [UNK]
            special_tokens = ["[PAD]", "[SEP]", "[EOS]", "[UNK]"]
        self.kkma = Kkma()
        self.special_tokens = special_tokens

        # Map from token(string) -> int, and reverse
        self.token2id = {}
        self.id2token = {}

    def build_vocab(self, texts):
        """
        Build vocabulary from a list of text strings.
        Each text is split into sentences, and each unique sentence is a token in the vocab.
        """
        # 1) Initialize with special tokens
        for i, t in enumerate(self.special_tokens):
            self.token2id[t] = i
        current_idx = len(self.special_tokens)

        # 2) Collect all sentences from the entire corpus
        sentence_set = set()
        for txt in texts:
            # Use kkma to split into sentences
            sents = self.kkma.sentences(txt)
            for s in sents:
                sentence_set.add(s.strip())

        # 3) Add each unique sentence to the vocabulary
        for s in sentence_set:
            if s not in self.token2id:
                self.token2id[s] = current_idx
                current_idx += 1

        # 4) Create reverse mapping
        self.id2token = {v: k for k, v in self.token2id.items()}

    def encode(self, text):
        """
        Encode a single text into a list of IDs (one ID per sentence).
        NOTE: For demonstration only – these "tokens" are entire sentences.
        """
        sents = self.kkma.sentences(text)
        ids = []
        for s in sents:
            s = s.strip()
            if s in self.token2id:
                ids.append(self.token2id[s])
            else:
                # Unknown sentence
                ids.append(self.token2id["[UNK]"])
        return ids

    def decode(self, ids):
        """
        Decode a list of integer IDs back to text (joining sentences with a space).
        """
        sents = []
        for idx in ids:
            if idx in self.id2token:
                sents.append(self.id2token[idx])
            else:
                sents.append("[UNK]")
        return " ".join(sents)

    @property
    def vocab_size(self):
        return len(self.token2id)

    def get_token_id(self, token="[PAD]"):
        """
        For convenience, returns the ID of a special token (e.g. [PAD], [SEP], [EOS]).
        """
        return self.token2id[token]

#####################################################
# 3. MambaLanguageModel 정의
#####################################################
class MambaLanguageModel(nn.Module):
    def __init__(self, vocab_size, d_model=16, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.mamba = Mamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand
        )
        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(self, input_ids):
        x = self.embedding(input_ids)  # [batch, seq_len, d_model]
        y = self.mamba(x)             # 동일 크기
        logits = self.lm_head(y)      # [batch, seq_len, vocab_size]
        return logits

#####################################################
# 4. 오토리그레시브(AR) 학습/추론 함수
#####################################################
def train_step(model, input_ids, optimizer, criterion):
    """1 step(training loop)에서 forward → backward → update까지 진행"""
    optimizer.zero_grad()
    logits = model(input_ids)
    # 예측(= t 시점) vs. 정답(= t+1 시점)
    preds = logits[:, :-1, :].contiguous()   # all but last token to predict next
    labels = input_ids[:, 1:].contiguous()   # all but first token as "labels"
    loss = criterion(preds.view(-1, preds.size(-1)), labels.view(-1))
    loss.backward()
    optimizer.step()
    return loss.item()

@torch.no_grad()
def val_step(model, input_ids, criterion):
    """검증 단계에서 loss 계산만 진행(역전파 X)"""
    logits = model(input_ids)
    preds = logits[:, :-1, :].contiguous()
    labels = input_ids[:, 1:].contiguous()
    loss = criterion(preds.view(-1, preds.size(-1)), labels.view(-1))
    return loss.item()

@torch.no_grad()
def generate_text(model, tokenizer, prompt, max_new_tokens=50, eos_token_id=None):
    """
    단순 Greedy Search로 텍스트 생성.
    Here each "token" is a sentence, so we generate one sentence at a time.
    """
    model.eval()

    # Encode prompt into token-IDs
    input_ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long).cuda()
    
    # Greedy decoding
    for _ in range(max_new_tokens):
        logits = model(input_ids)
        next_token_logits = logits[0, -1, :]        # last position
        next_token_id = torch.argmax(next_token_logits).unsqueeze(0).unsqueeze(0)
        input_ids = torch.cat([input_ids, next_token_id], dim=1)

        # Stop if we hit [EOS]
        if eos_token_id is not None and next_token_id.item() == eos_token_id:
            break

    # Decode back to text
    output_text = tokenizer.decode(input_ids[0].tolist())
    return output_text

#####################################################
# 5. 후처리 함수 (중복 "A:" 모두 처리)
#####################################################
def postprocess_answer(generated_text: str) -> str:
    """
    1) 모든 'A:' (대소문자 무관, 공백 무관) 패턴을 찾은 뒤
       마지막으로 등장하는 'A:' 위치 다음 텍스트만 추출
    2) [EOS] 제거
    3) 앞뒤 공백 제거
    """
    pattern = r"(?i)A\s*:\s*"
    matches = list(re.finditer(pattern, generated_text))
    
    if len(matches) == 0:
        # 'A:'가 전혀 없으면 전체 문장을 반환
        answer_part = generated_text
    else:
        # 마지막(match) A:의 끝 인덱스
        last_match = matches[-1]
        start_idx = last_match.end()
        answer_part = generated_text[start_idx:]
    
    # [EOS] 제거
    answer_part = answer_part.replace("[EOS]", "")
    # 앞뒤 공백 제거
    answer_part = answer_part.strip()
    
    return answer_part

#####################################################
# 6. KkmaTokenizer 준비 (vocab 빌드)
#####################################################

# 6-1. 수집할 전체 텍스트: Q + A 합치거나, 
#      필요하면 train/test 전부를 한꺼번에 처리
all_texts = []
for q, a in train_data:
    # For building vocab, combine Q, A, plus special tokens
    text_sequence = f"Q: {q} [SEP] A: {a} [EOS]"
    all_texts.append(text_sequence)

# (Optional) Also include test questions for coverage
for _, row in test.iterrows():
    q_text = (
        f"공종 중분류 '{row['공종(중분류)']}'에서 "
        f"작업 프로세스 '{row['작업프로세스']}' 와 관련된 사고가 발생했습니다. "
        f"사고 원인은 '{row['사고원인']}'입니다. 재발 방지 대책 및 향후 조치 계획은 무엇인가요? "
        "[SEP] A: [EOS]"
    )
    all_texts.append(q_text)

# Initialize our custom tokenizer
tokenizer = KkmaSentenceTokenizer()
tokenizer.build_vocab(all_texts)

print("Vocab size:", tokenizer.vocab_size)

#####################################################
# 7. MambaLanguageModel 초기화
#####################################################
mamba_lm = MambaLanguageModel(
    vocab_size=tokenizer.vocab_size,
    d_model=768,    # Example: BERT-base-like hidden size
    d_state=16,
    d_conv=4,
    expand=2
).cuda()

# 옵티마이저, 손실함수
optimizer = optim.AdamW(mamba_lm.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss()

#####################################################
# 8. Dataset, DataLoader (Train / Validation 분할)
#####################################################
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

class QADataset(Dataset):
    def __init__(self, pairs, tokenizer, max_length=512):
        """
        Each sample: "Q: ... [SEP] A: ... [EOS]"
        Then we encode via Kkma sentences and store the list of IDs.
        """
        self.samples = []
        self.tokenizer = tokenizer
        self.max_length = max_length

        for q, a in pairs:
            text_sequence = f"Q: {q} [SEP] A: {a} [EOS]"
            # Convert to IDs
            input_ids = self.tokenizer.encode(text_sequence)
            # Truncate if needed
            if len(input_ids) > self.max_length:
                input_ids = input_ids[:self.max_length]
            self.samples.append(torch.tensor(input_ids, dtype=torch.long))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

def collate_fn(batch):
    """가변 길이 시퀀스를 동일 길이로 패딩"""
    max_len = max(x.size(0) for x in batch)
    padded_batch = []
    pad_id = tokenizer.get_token_id("[PAD]")
    for x in batch:
        pad_size = max_len - x.size(0)
        padded = torch.cat([x, torch.full((pad_size,), pad_id, dtype=torch.long)])
        padded_batch.append(padded.unsqueeze(0))
    return torch.cat(padded_batch, dim=0)

# 8-1. Train / Val 분할
train_pairs, val_pairs = train_test_split(train_data, test_size=0.2, random_state=42)
train_dataset = QADataset(train_pairs, tokenizer, max_length=512)
val_dataset   = QADataset(val_pairs, tokenizer, max_length=512)

train_loader = DataLoader(
    train_dataset,
    batch_size=8,
    shuffle=True,
    collate_fn=collate_fn
)
val_loader = DataLoader(
    val_dataset,
    batch_size=8,
    shuffle=False,
    collate_fn=collate_fn
)

#####################################################
# 9. Early Stopping을 고려한 학습 루프
#####################################################
num_epochs = 10  # For demo, smaller epoch count
patience = 3
best_val_loss = float("inf")
counter = 0

for epoch in range(num_epochs):
    # === [훈련 단계] ===
    mamba_lm.train()
    total_train_loss = 0.0

    for step, input_ids in enumerate(train_loader):
        input_ids = input_ids.cuda()
        loss_val = train_step(mamba_lm, input_ids, optimizer, criterion)
        total_train_loss += loss_val

        if (step + 1) % 100 == 0:
            avg_train_loss = total_train_loss / (step + 1)
            print(f"[Epoch {epoch+1}, Step {step+1}] train_loss={avg_train_loss:.4f}")

    # === [검증 단계] ===
    mamba_lm.eval()
    total_val_loss = 0.0
    for val_input_ids in val_loader:
        val_input_ids = val_input_ids.cuda()
        val_loss_val = val_step(mamba_lm, val_input_ids, criterion)
        total_val_loss += val_loss_val
    
    avg_val_loss = total_val_loss / len(val_loader)
    avg_train_loss = total_train_loss / len(train_loader)
    print(f"=== Epoch {epoch+1} / {num_epochs} ===")
    print(f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

    # Early Stopping 체크
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        counter = 0
        print("[Info] Validation loss improved. Model saved.\n")
        # torch.save(mamba_lm.state_dict(), "best_mamba_model.pt")
    else:
        counter += 1
        print(f"[Info] Validation loss did NOT improve. Counter={counter}/{patience}\n")
        if counter >= patience:
            print("[Early Stopping] No improvement after {} epochs. Training stopped.".format(patience))
            break

print("최종 학습 완료!")

#####################################################
# 10. Mamba + LangChain 연동 (선택)
#####################################################
class MambaLLM(LLM):
    model_name: str = "Mamba"

    def __init__(
        self,
        model: nn.Module,
        tokenizer: KkmaSentenceTokenizer,
        max_new_tokens: int = 64,
        **kwargs
    ):
        super().__init__(**kwargs)
        self._model = model
        self._tokenizer = tokenizer
        self._max_new_tokens = max_new_tokens
        # Let's store the [EOS] token ID
        self._eos_token_id = tokenizer.get_token_id("[EOS]")

    @property
    def _llm_type(self) -> str:
        return "mamba_custom"

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        output = generate_text(
            model=self._model,
            tokenizer=self._tokenizer,
            prompt=prompt,
            max_new_tokens=self._max_new_tokens,
            eos_token_id=self._eos_token_id
        )
        return output

#####################################################
# 11. test.csv 예측 후 submission.csv 생성
#####################################################
test_results = []

# If you still want to embed the final answers, you can 
# re-introduce a vector embedding approach here. (Optional)
# embedding_model = HuggingFaceEmbeddings(model_name="jhgan/ko-sbert-nli")

for i, row in test.iterrows():
    question_text = (
        f"Q: 공종 중분류 '{row['공종(중분류)']}'에서 "
        f"작업 프로세스 '{row['작업프로세스']}' 와 관련된 사고가 발생했습니다. "
        f"사고 원인은 '{row['사고원인']}'입니다. 재발 방지 대책 및 향후 조치 계획은 무엇인가요? "
        "[SEP] A:"
    )

    generated_answer = generate_text(
        model=mamba_lm,
        tokenizer=tokenizer,
        prompt=question_text,
        max_new_tokens=50,  # for example
        eos_token_id=tokenizer.get_token_id("[EOS]")
    )

    # 후처리: "마지막 A:" 뒤의 텍스트만 추출
    clean_answer = postprocess_answer(generated_answer)

    # (Optional) 임베딩
    # answer_vector = embedding_model.embed_query(clean_answer)

    record = {
        "ID": row["ID"],
        "재발방지대책 및 향후조치계획": clean_answer
    }
    # 예시: 벡터를 CSV에 넣으려면
    # for idx_dim in range(len(answer_vector)):
    #     record[f"vec_{idx_dim}"] = answer_vector[idx_dim]

    test_results.append(record)

submission_df = pd.DataFrame(test_results)
submission_df.to_csv("submission.csv", index=False, encoding="utf-8-sig")
print("submission.csv 파일 생성 완료!")
