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
# 1. 데이터 로드 및 전처리
############################################
train = pd.read_csv("./open/train.csv", encoding="utf-8-sig")
test = pd.read_csv("./open/test.csv", encoding="utf-8-sig")

# 예시: 질문/답변 형식으로 만들기 위해 전처리
# (train.csv 내 각 row에서 'question', 'answer' 컬럼 구성)
# 사용자 CSV 구조에 맞춰 수정하세요.

# 공사종류(대분류/중분류), 공종(대분류/중분류), 사고객체(대/중), 작업프로세스, 사고원인, 재발방지대책 등
train["공사종류(대분류)"] = train["공사종류"].str.split(" / ").str[0]
train["공사종류(중분류)"] = train["공사종류"].str.split(" / ").str[1]
train["공종(대분류)"] = train["공종"].str.split(" > ").str[0]
train["공종(중분류)"] = train["공종"].str.split(" > ").str[1]
train["사고객체(대분류)"] = train["사고객체"].str.split(" > ").str[0]
train["사고객체(중분류)"] = train["사고객체"].str.split(" > ").str[1]

train_data = []
for _, row in train.iterrows():
    q = (
        f"공사종류 대분류 '{row['공사종류(대분류)']}', "
        f"중분류 '{row['공사종류(중분류)']}' 공사 중 "
        f"공종 대분류 '{row['공종(대분류)']}', "
        f"중분류 '{row['공종(중분류)']}' 작업에서 "
        f"사고객체 '{row['사고객체(대분류)']}'(중분류: '{row['사고객체(중분류)']}')와 관련된 사고가 발생했습니다. "
        f"작업 프로세스는 '{row['작업프로세스']}'이며, "
        f"사고 원인은 '{row['사고원인']}'입니다. 재발 방지 대책 및 향후 조치 계획은 무엇인가요?"
    )
    a = row["재발방지대책 및 향후조치계획"]
    train_data.append((q, a))

# test.csv도 필요하다면 유사하게 전처리
# (다만 여기서는 파인튜닝 예시에 초점)
# test["공사종류(대분류)"] = ...
# ...

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
        y = self.mamba(x)             # 동일 크기
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
    # preds: [batch, seq_len-1, vocab_size]
    # labels: [batch, seq_len-1]
    preds = logits[:, :-1, :].contiguous()
    labels = input_ids[:, 1:].contiguous()

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
        # 전체 시퀀스에 대한 로짓
        logits = model(input_ids)  # [1, seq_len, vocab_size]
        next_token_logits = logits[0, -1, :]  # 마지막 토큰 위치 로짓 [vocab_size]
        next_token_id = torch.argmax(next_token_logits).unsqueeze(0).unsqueeze(0)  # [1,1]
        input_ids = torch.cat([input_ids, next_token_id], dim=1)

        if eos_token_id is not None and next_token_id.item() == eos_token_id:
            break

    # 최종 토큰 시퀀스를 디코딩
    output_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
    return output_text

############################################
# 4. Tokenizer 설정 (한국어 전용 권장)
############################################
# 여기서는 예시로 'klue/bert-base' 사용 (한국어 전용)
# huggingface에서 다른 ko 모델도 가능 (kobert, koelectra 등)
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
    d_model=32,   # 임베딩 차원 등은 상황에 맞게
    d_state=16,
    d_conv=4,
    expand=2
).cuda()

# 옵티마이저, 손실함수
optimizer = optim.AdamW(mamba_lm.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss()

############################################
# 6. 간단한 파인튜닝 루프 예시
############################################
# (실제로는 DataLoader, 배치 처리, 여러 epoch 등 구현 필요)
train_examples = []

for q, a in train_data:
    # "[Q: ...] [SEP] [A: ...] [EOS]" 형식으로 하나의 문자열 구성
    # 자유롭게 포맷 가능
    text_sequence = f"Q: {q} [SEP] A: {a} [EOS]"
    enc = tokenizer(text_sequence, return_tensors="pt")
    train_examples.append(enc["input_ids"][0])  # [seq_len]

# 패딩, 배치 처리를 위해 간단한 예시로 전부 최대길이에 맞춰 pad
max_length = max(x.size(0) for x in train_examples)
padded_inputs = []
for ids in train_examples:
    pad_size = max_length - ids.size(0)
    padded = torch.cat([ids, torch.full((pad_size,), tokenizer.pad_token_id)])
    padded_inputs.append(padded.unsqueeze(0))

train_tensor = torch.cat(padded_inputs, dim=0).cuda()  # [N, max_length]

print("Train shape:", train_tensor.shape)  # (샘플 수, max_length)

# 간단히 1 에포크만 예시
for step in range(10):  # 실제로는 수백~수천 스텝
    loss_val = train_step(mamba_lm, train_tensor, optimizer, criterion)
    if (step + 1) % 1 == 0:
        print(f"Step {step+1}, loss={loss_val:.4f}")

############################################
# 7. (선택) Mamba + LangChain 연동
############################################
# LangChain은 일반적으로 LLM(질문 + 컨텍스트 → 답변)을 호출해야 합니다.
# Mamba는 HF pipeline 형태가 아니므로, 커스텀 LLM 클래스를 만들어 _call()에서 generate_text() 수행

class MambaLLM(LLM):
    def __init__(self, model: nn.Module, tokenizer, max_new_tokens=64):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.max_new_tokens = max_new_tokens
        # eos_token_id 설정 가능
        self.eos_token_id = tokenizer.sep_token_id  # 예시

    @property
    def _llm_type(self) -> str:
        return "mamba_custom"

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        # prompt가 들어오면 Mamba 기반으로 텍스트를 생성
        output = generate_text(
            model=self.model,
            tokenizer=self.tokenizer,
            prompt=prompt,
            max_new_tokens=self.max_new_tokens,
            eos_token_id=self.eos_token_id
        )
        return output

# (1) 벡터스토어(RAG) 준비
# 예: train_documents를 FAISS에 넣는다.
from langchain.embeddings import HuggingFaceEmbeddings
embedding_model_name = "jhgan/ko-sbert-nli"
emb = HuggingFaceEmbeddings(model_name=embedding_model_name)

# Q: ...\nA: ... 형태로 문서를 구성할 수도 있음
train_documents = []
for q, a in train_data:
    doc_text = f"Q: {q}\nA: {a}"
    train_documents.append(doc_text)

vector_store = FAISS.from_texts(train_documents, emb)
retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 5})

# (2) LangChain PromptTemplate
prompt_template = """
당신은 건설 안전 전문가입니다. 다음 정보를 참고해 간략히 답변을 제시하세요.
{context}

질문: {question}
""".strip()

my_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=prompt_template
)

# (3) RetrievalQA 체인
mamba_llm_wrapper = MambaLLM(mamba_lm, tokenizer, max_new_tokens=64)

qa_chain = RetrievalQA.from_chain_type(
    llm=mamba_llm_wrapper,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True,
    chain_type_kwargs={"prompt": my_prompt},
)

# (4) 테스트
test_question = "크레인 작업 시 필요한 안전조치는?"
result = qa_chain.run(test_question)
print("질문:", test_question)
print("모델 답변:", result)


############################################
# 참고: 실제로는 Mamba가 충분한 데이터로 사전학습되지 않았으므로
#       답변이 제대로 나오기 어려울 수 있음.
############################################

