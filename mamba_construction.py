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

# 예시: 질문/답변 형식으로 만들기 위해 전처리
train["공사종류(대분류)"] = train["공사종류"].str.split(" / ").str[0]
train["공사종류(중분류)"] = train["공사종류"].str.split(" / ").str[1]
train["공종(대분류)"] = train["공종"].str.split(" > ").str[0]
train["공종(중분류)"] = train["공종"].str.split(" > ").str[1]
train["사고객체(대분류)"] = train["사고객체"].str.split(" > ").str[0]
train["사고객체(중분류)"] = train["사고객체"].str.split(" > ").str[1]

train_data = []
for _, row in train.iterrows():
    q = (
        f"공종 중분류 '{row['공종(중분류)']}'에서 "
        f"작업 프로세스 '{row['작업프로세스']}' 와 관련된 사고가 발생했습니다."
        f"사고 원인은 '{row['사고원인']}'입니다. 재발 방지 대책 및 향후 조치 계획은 무엇인가요?"
    )
    a = row["재발방지대책 및 향후조치계획"]  # 정답(대책)
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
        """
        input_ids: [batch, seq_len]
        반환: logits [batch, seq_len, vocab_size]
        """
        x = self.embedding(input_ids)  # [batch, seq_len, d_model]
        y = self.mamba(x)             # 동일 크기 [batch, seq_len, d_model]
        logits = self.lm_head(y)      # [batch, seq_len, vocab_size]
        return logits

############################################
# 3. 오토리그레시브(AR) 학습 함수 및 추론 함수
############################################
def train_step(model, input_ids, optimizer, criterion):
    """
    한 스텝 학습 (오토리그레시브 LM 방식)
    input_ids: [batch, seq_len] (Q + A)
    예) "[Q: ...] [SEP] [A: ...] [EOS]" 전체를 하나의 시퀀스로.
    """
    optimizer.zero_grad()
    logits = model(input_ids)  # [batch, seq_len, vocab_size]

    # Shift tokens by 1 for next-token prediction
    preds = logits[:, :-1, :].contiguous()  # [batch, seq_len-1, vocab_size]
    labels = input_ids[:, 1:].contiguous()  # [batch, seq_len-1]

    loss = criterion(preds.view(-1, preds.size(-1)), labels.view(-1))
    loss.backward()
    optimizer.step()
    return loss.item()

@torch.no_grad()
def generate_text(model, tokenizer, prompt, max_new_tokens=50, eos_token_id=None):
    """
    간단한 Greedy 디코딩 예시
    model: MambaLanguageModel
    tokenizer: HF tokenizer
    prompt: str
    """
    model.eval()
    input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].cuda()  # [1, seq_len]

    for _ in range(max_new_tokens):
        logits = model(input_ids)  # [1, seq_len, vocab_size]
        next_token_logits = logits[0, -1, :]  # 마지막 토큰의 로짓
        next_token_id = torch.argmax(next_token_logits).unsqueeze(0).unsqueeze(0)  # [1,1]
        input_ids = torch.cat([input_ids, next_token_id], dim=1)

        if eos_token_id is not None and next_token_id.item() == eos_token_id:
            break

    output_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
    return output_text

############################################
# 4. Tokenizer 설정 (한국어 전용 권장)
############################################
tokenizer_name = "klue/bert-base"
tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
vocab_size = tokenizer.vocab_size

# EOS 토큰이 명시돼 있지 않다면, tokenizer의 sep_token_id 등을 임시로 사용 가능
# tokenizer.eos_token_id = tokenizer.sep_token_id

############################################
# 5. MambaLanguageModel 초기화
############################################
mamba_lm = MambaLanguageModel(
    vocab_size=vocab_size,
    d_model=768,  # 여기서 d_model=32 -> 임베딩 벡터 차원 32
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
    """단순 (Q, A) 튜플 목록 -> 토큰 시퀀스 텐서"""
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
            input_ids = enc["input_ids"].squeeze(0)  # shape: [seq_len]
            self.samples.append(input_ids)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

def collate_fn(batch):
    """가변 길이 배치를 pad하여 텐서화"""
    max_len = max(x.size(0) for x in batch)
    padded_batch = []
    for x in batch:
        pad_size = max_len - x.size(0)
        padded = torch.cat([x, torch.full((pad_size,), tokenizer.pad_token_id, dtype=torch.long)])
        padded_batch.append(padded.unsqueeze(0))
    return torch.cat(padded_batch, dim=0)  # [batch_size, max_len_in_batch]

# 학습용 데이터셋/데이터로더
train_dataset = QADataset(train_data, tokenizer, max_length=512)
train_loader = DataLoader(
    train_dataset,
    batch_size=8,
    shuffle=True,
    collate_fn=collate_fn
)

# 간단히 2 에폭 학습
num_epochs = 2
for epoch in range(num_epochs):
    for step, input_ids in enumerate(train_loader):
        input_ids = input_ids.cuda()
        loss_val = train_step(mamba_lm, input_ids, optimizer, criterion)

        if (step + 1) % 100 == 0:
            print(f"Epoch {epoch+1}, Step {step+1}, Loss={loss_val:.4f}")

############################################
# 7. (선택) LangChain과 연동 (예시)
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
        self._eos_token_id = tokenizer.sep_token_id  # or define a real EOS

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

# 벡터스토어 & RetrievalQA 예시
# (주의: 실제로는 파인튜닝 된 Mamba 성능이 충분치 않을 수 있음)
embedding_model_name = "jhgan/ko-sbert-nli"
emb = HuggingFaceEmbeddings(model_name=embedding_model_name)

train_documents = []
for q, a in train_data:
    doc_text = f"Q: {q}\nA: {a}"
    train_documents.append(doc_text)

vector_store = FAISS.from_texts(train_documents, emb)
retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 5})

prompt_template = """
당신은 건설 안전 전문가입니다. 다음 정보를 참고해 간략히 답변을 제시하세요.
{context}

질문: {question}
""".strip()

my_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=prompt_template
)

mamba_llm_wrapper = MambaLLM(mamba_lm, tokenizer, max_new_tokens=64)
qa_chain = RetrievalQA.from_chain_type(
    llm=mamba_llm_wrapper,
    chain_type="stuff",
    retriever=retriever,
    chain_type_kwargs={"prompt": my_prompt},
)

# test_question = "크레인 작업 시 필요한 안전조치는?"
# result = qa_chain.run(test_question)
# print("질문:", test_question)
# print("모델 답변:", result)

############################################
# 8. TEST 데이터셋에 대한 임베딩/벡터 추출 -> submission.csv
############################################

# (1) 테스트셋 로드 및 전처리
test = pd.read_csv("./open/test.csv", encoding="utf-8-sig")

# train과 동일하게 전처리
test["공사종류(대분류)"] = test["공사종류"].str.split(" / ").str[0]
test["공사종류(중분류)"] = test["공사종류"].str.split(" / ").str[1]
test["공종(대분류)"] = test["공종"].str.split(" > ").str[0]
test["공종(중분류)"] = test["공종"].str.split(" > ").str[1]
test["사고객체(대분류)"] = test["사고객체"].str.split(" > ").str[0]
test["사고객체(중분류)"] = test["사고객체"].str.split(" > ").str[1]

# (2) "Q: ... [SEP] A:" 형태로 질문만 생성 (답변은 없음)
# test에는 '재발방지대책' 컬럼이 없으므로, 답변 없이 Q만 생성
test_data = []
for idx, row in test.iterrows():
    # 필요 시 test셋에 고유 ID가 있으면 그것을 사용
    # 여기서는 예시로 "TEST_000" 같은 식으로 붙임
    sample_id = f"TEST_{idx:03d}"

    q = (
        f"공종 중분류 '{row['공종(중분류)']}'에서 "
        f"작업 프로세스 '{row['작업프로세스']}' 와 관련된 사고가 발생했습니다."
        f"사고 원인은 '{row['사고원인']}'입니다. 재발 방지 대책 및 향후 조치 계획은 무엇인가요?"
    )
    # '작업프로세스'를 두 번째 컬럼 예시로 활용(원하시는 컬럼 넣어도 됨)
    test_data.append((sample_id, row["작업프로세스"], q))

# (3) 모델의 임베딩(마지막 hidden state) 추출 함수
@torch.no_grad()
def encode_sentence(model, tokenizer, text):
    """
    text(문자열)을 입력받아 model의 마지막 hidden state(예: 마지막 토큰)를 벡터로 추출
    d_model=32이면 shape가 32차원이 나옴
    """
    model.eval()
    input_ids = tokenizer(text, return_tensors="pt")["input_ids"].cuda()  # [1, seq_len]
    x = model.embedding(input_ids)       # [1, seq_len, d_model]
    y = model.mamba(x)                  # [1, seq_len, d_model]
    last_hidden = y[:, -1, :]          # 마지막 토큰의 hidden state -> [1, d_model]
    vec = last_hidden.squeeze(0).cpu()  # shape: [d_model]
    return vec  # tensor of shape (d_model,)

# (4) 테스트셋 각 샘플에 대해 벡터화 & submission.csv로 저장
submission_rows = []
for sample_id, process, question_prompt in test_data:
    # "Q: {prompt} [SEP] A:" 형태로 입력 (train 대비 답변은 없음)
    text_input = f"Q: {question_prompt} [SEP] A:"
    vec = encode_sentence(mamba_lm, tokenizer, text_input)

    # 텐서를 파이썬 리스트로 변환
    vec_list = vec.tolist()

    # CSV 각 행: [ID, 작업프로세스, dim1, dim2, ..., dimN]
    row_data = [sample_id, process] + vec_list
    submission_rows.append(row_data)

# 컬럼 이름 (원하시는 형태에 맞게 수정 가능)
dim_size = len(vec_list)  # 예: 32
columns = ["ID", "작업프로세스"] + [f"dim_{i}" for i in range(dim_size)]

# 데이터프레임 생성
df_submit = pd.DataFrame(submission_rows, columns=columns)

# CSV로 저장 (인덱스 없이)
df_submit.to_csv("submission.csv", index=False)

print("Submission file (submission.csv) was created.")


############################################
# 9. test 첫 번째 물음에 대한 답변 직접 확인
############################################

# test_data: [(sample_id, process, question_prompt), ...]

# 1) 첫 번째 샘플의 question_prompt 가져오기
first_sample_id, first_process, first_question_prompt = test_data[0]

# 2) 모델에 직접 넣어서 답변 생성 (generate_text 버전)
#    train.csv와 똑같이 "Q: ... [SEP] A:" 형태 입력
input_text = f"Q: {first_question_prompt} [SEP] A:"
print("=== [직접 generate_text] ===")
print("입력프롬프트:", input_text)
answer_direct = generate_text(
    model=mamba_lm,
    tokenizer=tokenizer,
    prompt=input_text,
    max_new_tokens=100,  # 필요 시 조절
    eos_token_id=tokenizer.sep_token_id
)
print("모델 답변:", answer_direct, "\n")

# 3) LangChain RetrievalQA 체인을 통해 답변 생성
#    test 질문을 그대로 qa_chain.run()에 넣으면,
#    내부적으로 벡터스토어 검색 + Mamba LLM으로 답변.
#    (주의: prompt_template와 결합되며, "질문: {question}" 형식으로 들어감)
print("=== [RetrievalQA] ===")
print("질문:", first_question_prompt)
answer_rqa = qa_chain.run(first_question_prompt)
print("모델 답변 (RetrievalQA):", answer_rqa)

