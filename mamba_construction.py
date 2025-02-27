import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd

# Mamba 관련 라이브러리
from mamba_ssm import Mamba

# Hugging Face 및 LangChain 관련
from transformers import AutoTokenizer
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.llms.base import LLM
from typing import Optional, List

############################################
# 1. 데이터 로드 및 전처리 (train.csv)
############################################
train = pd.read_csv("./open/train.csv", encoding="utf-8-sig")

train["공사종류(대분류)"] = train["공사종류"].str.split(" / ").str[0]
train["공사종류(중분류)"] = train["공사종류"].str.split(" / ").str[1]
train["공종(대분류)"] = train["공종"].str.split(" > ").str[0]
train["공종(중분류)"] = train["공종"].str.split(" > ").str[1]
train["사고객체(대분류)"] = train["사고객체"].str.split(" > ").str[0]
train["사고객체(중분류)"] = train["사고객체"].str.split(" > ").str[1]

# (Q, A) 튜플 리스트 생성
train_data = []
for _, row in train.iterrows():
    q = (
        f"공종 중분류 '{row['공종(중분류)']}'에서 "
        f"작업 프로세스 '{row['작업프로세스']}' 와 관련된 사고가 발생했습니다."
        f"사고 원인은 '{row['사고원인']}'입니다. 재발 방지 대책 및 향후 조치 계획은 무엇인가요?"
    )
    a = row["재발방지대책 및 향후조치계획"]
    train_data.append((q, a))

############################################
# 2. 시퀀스(문장) 생성 모델 (Mamba + LM Head) 정의
############################################
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

############################################
# 3. 오토리그레시브(AR) 학습 함수 및 추론 함수
############################################
def train_step(model, input_ids, optimizer, criterion):
    optimizer.zero_grad()
    logits = model(input_ids)
    preds = logits[:, :-1, :].contiguous()
    labels = input_ids[:, 1:].contiguous()
    loss = criterion(preds.view(-1, preds.size(-1)), labels.view(-1))
    loss.backward()
    optimizer.step()
    return loss.item()

@torch.no_grad()
def generate_text(model, tokenizer, prompt, max_new_tokens=50, eos_token_id=None):
    model.eval()
    input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].cuda()

    for _ in range(max_new_tokens):
        logits = model(input_ids)
        next_token_logits = logits[0, -1, :]
        next_token_id = torch.argmax(next_token_logits).unsqueeze(0).unsqueeze(0)
        input_ids = torch.cat([input_ids, next_token_id], dim=1)

        if eos_token_id is not None and next_token_id.item() == eos_token_id:
            break

    output_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
    return output_text

####################################################
# (NEW) Postprocessing function to cut off "A:" part
####################################################
def postprocess_answer(generated_text: str) -> str:
    """
    Extract only the portion after the first 'A:' marker,
    and remove [EOS] or extra whitespace if present.
    """
    marker = "A:"
    idx = generated_text.find(marker)
    
    if idx != -1:
        # substring after 'A:'
        answer_part = generated_text[idx + len(marker):]
    else:
        # if no 'A:' found, return the entire text or an empty string
        answer_part = generated_text

    # remove any [EOS] tokens
    answer_part = answer_part.replace("[EOS]", "")
    # strip extra whitespace
    answer_part = answer_part.strip()

    return answer_part

############################################
# 4. Tokenizer 설정 (한국어 전용 권장)
############################################
tokenizer_name = "klue/bert-base"
tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
vocab_size = tokenizer.vocab_size

############################################
# 5. MambaLanguageModel 초기화
############################################
mamba_lm = MambaLanguageModel(
    vocab_size=vocab_size,
    d_model=768,
    d_state=16,
    d_conv=4,
    expand=2
).cuda()

# 옵티마이저, 손실함수
optimizer = optim.AdamW(mamba_lm.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss()

############################################
# 6. DataLoader로 미니배치 학습
############################################
from torch.utils.data import Dataset, DataLoader

class QADataset(Dataset):
    def __init__(self, pairs, tokenizer, max_length=512):
        self.samples = []
        self.tokenizer = tokenizer
        self.max_length = max_length

        for q, a in pairs:
            text_sequence = f"Q: {q} [SEP] A: {a} [EOS]"
            enc = self.tokenizer(
                text_sequence,
                return_tensors="pt",
                truncation=True,
                max_length=self.max_length
            )
            input_ids = enc["input_ids"].squeeze(0)
            self.samples.append(input_ids)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

def collate_fn(batch):
    max_len = max(x.size(0) for x in batch)
    padded_batch = []
    for x in batch:
        pad_size = max_len - x.size(0)
        padded = torch.cat([x, torch.full((pad_size,), tokenizer.pad_token_id, dtype=torch.long)])
        padded_batch.append(padded.unsqueeze(0))
    return torch.cat(padded_batch, dim=0)

train_dataset = QADataset(train_data, tokenizer, max_length=512)
train_loader = DataLoader(
    train_dataset,
    batch_size=8,
    shuffle=True,
    collate_fn=collate_fn
)

num_epochs = 30
for epoch in range(num_epochs):
    for step, input_ids in enumerate(train_loader):
        input_ids = input_ids.cuda()
        loss_val = train_step(mamba_lm, input_ids, optimizer, criterion)

        if (step + 1) % 100 == 0:
            print(f"Epoch {epoch+1}, Step {step+1}, Loss={loss_val:.4f}")

############################################
# 7. (선택) Mamba + LangChain 연동 (예시)
############################################
class MambaLLM(LLM):
    model_name: str = "Mamba"

    def __init__(
        self,
        model: nn.Module,
        tokenizer,
        max_new_tokens: int = 64,
        **kwargs
    ):
        super().__init__(**kwargs)
        self._model = model
        self._tokenizer = tokenizer
        self._max_new_tokens = max_new_tokens
        self._eos_token_id = tokenizer.sep_token_id

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

############################################
# 8. test.csv 예측 후 submission.csv 생성
############################################
test = pd.read_csv("./open/test.csv", encoding="utf-8-sig")
test["공사종류(대분류)"] = test["공사종류"].str.split(" / ").str[0]
test["공사종류(중분류)"] = test["공사종류"].str.split(" / ").str[1]
test["공종(대분류)"] = test["공종"].str.split(" > ").str[0]
test["공종(중분류)"] = test["공종"].str.split(" > ").str[1]
test["사고객체(대분류)"] = test["사고객체"].str.split(" > ").str[0]
test["사고객체(중분류)"] = test["사고객체"].str.split(" > ").str[1]

# 만약 test.csv에 ID가 없으면 생성 가능
# test["ID"] = test.index.map(lambda x: f"TEST_{x:03d}")

embedding_model = HuggingFaceEmbeddings(model_name="jhgan/ko-sbert-nli")

test_results = []

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
        max_new_tokens=100,
        eos_token_id=tokenizer.sep_token_id
    )

    ####################################################
    # (NEW) Apply postprocessing so we keep only the part
    #       after "A:"
    ####################################################
    clean_answer = postprocess_answer(generated_answer)

    answer_vector = embedding_model.embed_query(clean_answer)

    record = {
        "ID": row["ID"],
        # store the clean answer in the CSV
        "재발방지대책 및 향후조치계획": clean_answer
    }
    for idx_dim in range(len(answer_vector)):
        record[f"vec_{idx_dim}"] = answer_vector[idx_dim]

    test_results.append(record)

submission_df = pd.DataFrame(test_results)
submission_df.to_csv("submission.csv", index=False)

print("submission.csv 파일 생성 완료!")
