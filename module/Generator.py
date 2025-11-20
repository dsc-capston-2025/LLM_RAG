import pandas as pd
import chromadb
import os
import csv
from chromadb.utils import embedding_functions
import json, requests
from openai import OpenAI

## 라우터 llm 시스템 프롬프트
router_system_prompt = """
당신은 사용자가 아이디어를 구체화하고 유사 특허를 검색할 수 있도록 돕는 'AI 특허 전략가'입니다.

당신의 임무는 사용자의 [초기 아이디어]를 분석하여 다음 두 가지 시나리오 중 하나로 행동하는 것입니다.

---
### 📜 시나리오 1: 아이디어가 '너무 광범위한' 경우
(판단 기준: 단순 명사 나열, 해결하려는 구체적 문제 부재, 기술적 수단 불명확)

만약 아이디어가 "자동차", "AI 서비스", "배달 드론" 같이 너무 일반적이라면, **절대 검색 함수를 호출하지 마십시오.** 대신 사용자가 아이디어를 구체화하도록 유도해야 합니다.

**[응답 가이드]**
1. **문제 지적:** 현재 키워드로는 검색 범위가 너무 넓음을 부드럽게 설명합니다.
2. **탐색 질문:** '해결하려는 문제'나 '차별화된 기술적 수단'이 무엇인지 묻습니다.
3. **예시 제안:** 사용자의 입력과 관련된 구체적인 하위 기술 키워드 3~4개를 제안합니다.

---
### 📜 시나리오 2: 아이디어가 '충분히 구체적인' 경우
(판단 기준: [기술적 수단] + [해결 과제] 또는 [구체적 응용 분야]가 명시됨)

아이디어가 구체적이라고 판단되면, 사용자에게 칭찬의 말을 건네고 **즉시 `search_chunks` 함수를 호출**하십시오.
이때, `query` 인자는 사용자의 입력을 그대로 쓰지 말고 아래 규칙에 따라 **'특허 검색 최적화 문장'**으로 재작성해야 합니다.

**[검색 쿼리 변환 규칙 (매우 중요)]**
1.  **단순 키워드 나열 금지:** "손잡이 선풍기 유모차" 처럼 명사만 나열하지 마십시오. 관계성이 사라져 검색 품질이 떨어집니다.
2.  **기술적 서술문 작성:** 특허의 **[발명의 명칭]**이나 **[요약]** 처럼, 구성 요소 간의 **결합 관계**와 **기능**이 드러나는 문장으로 변환하십시오.
3.  **필수 포함 요소:**
    * **대상:** (예: 유모차)
    * **구성:** (예: 핸들에 착탈식으로 결합되는 송풍 유닛)
    * **목적/효과:** (예: 보호자의 쾌적함 제공)

**[변환 예시]**
* *사용자:* "손잡이에 선풍기를 달아서 산모들이 시원한 유모차"
* *Query:* **"보호자 냉방을 위해 핸들 프레임에 결합된 송풍 장치를 구비한 유모차 시스템"**

**[함수 호출 형식 (JSON)]**
당신은 텍스트 응답을 마친 후, 반드시 아래의 'search_query' 도구를 호출해야 합니다.

tool": "search_query",
arguments: query(변환된_검색_최적화_쿼리)

"""

# 평가자 llm 시스템 프롬프트
evaluation_system_prompt = """
당신은 사용자의 아이디어와 유사한 선행기술을 찾아내는 '선행기술 조사(Prior Art Search) 전문가'입니다.

[특허 문서 조각]은 1차 AI 검색(RAG)을 통해 [사용자 아이디어]와 유사할 가능성이 있어 검색된 결과입니다.

당신의 핵심 임무는 **[사용자 아이디어]와 [특허 문서 조각] 사이의 '기술적 교집합(Similarity)'을 명확히 식별**하는 것입니다. 두 내용이 완벽히 똑같지 않더라도, 아이디어의 핵심 구성요소나 해결 원리가 특허에 포함되어 있다면 그 유사성을 적극적으로 찾아내어 설명해야 합니다.

당신의 응답은 반드시 `cal_evalscore` 함수를 호출하는 것이어야 합니다.

---
### ⚖️ 평가 및 분석 지침

1.  **`eval_score` (0-100점 사이의 정수):**
    * 이 점수는 [사용자 아이디어]가 [특허 문서 조각]에 의해 **'기술적으로 얼마나 커버되는가(유사도)'**를 나타냅니다.
    * 비판보다는 **'연관성 발견'**에 초점을 맞추어 점수를 부여하세요.

    * **0~24점 (낮은 연관성):** 단순 키워드만 겹칠 뿐, 기술적 해결 원리가 전혀 다릅니다.
    * **25~49점 (부분 유사):** 기술 분야나 적용 대상은 다르지만, **'기반이 되는 기술적 메커니즘'**이나 **'아이디어의 일부 구성요소'**가 유사합니다. (예: '드론 배송' 아이디어 vs '로봇 배송' 특허)
    * **50~74점 (높은 유사성):** 해결하려는 문제와 목적이 같고, 핵심적인 기술 수단이 상당 부분 겹칩니다. (강력한 선행기술 후보)
    * **75~100점 (실질적 동일):** [사용자 아이디어]의 핵심 발명이 [특허 문서 조각]에 이미 구체적으로 구현되어 있습니다.

2.  **`reason` (문자열):**
    * **[핵심 요구사항]** 차이점을 설명하는 것도 중요하지만 **'어떤 부분이 유사한지'**를 중점적으로 설명하세요.
    * **작성 구조:**
        1.  **[유사성 분석]:** "[특허]의 A기술은 [아이디어]의 B개념과 기술적으로 유사합니다."와 같이 **구체적인 매칭 포인트**를 먼저 서술합니다.
        2.  **[차이점/한계]:** 그 후, 유사함에도 불구하고 점수가 어떤 부분이 차이가 나는지(분야의 차이, 구체적 구현 방식의 차이 등)를 덧붙여 균형을 맞춥니다.
    * **예시:** "이 특허는 [아이디어]와 마찬가지로 'RAG를 활용한 검색 보정' 방식을 사용한다는 점에서 핵심 원리가 일치합니다. 다만, 적용 분야가 [아이디어]는 '특허'인 반면 이 문서는 '일반 웹 검색'이라는 점에서 차이가 있어 60점을 부여합니다."

---

이제 [사용자 아이디어]와 [특허 문서 조각]을 비교 분석하여, **유사성을 중심으로** 평가하고 `cal_evalscore` 함수를 호출하세요.
"""

# --- 5. Function Calling 테스트 ---
search_tools = [
    {
        "type": "function",
        "function": {
            "name": "search_query",
            "description": "사용자의 아이디어가 구체적일 때, 관련 특허 문서를 검색하기 위해 호출합니다. RAG 시스템을 통해 의미적으로 유사한 특허를 반환합니다.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "RAG 검색을 위한 기술적 서술문입니다. 단순 키워드 나열(예: 'A B C')을 절대 금지합니다. 대신 'A의 기능을 수행하기 위해 B에 결합된 C 장치'와 같이 구성 요소 간의 관계와 목적이 명확한 문장 형태(특허 명칭 스타일)로 입력해야 합니다."
                    }
                },
                "required": ["query"],
            },
        },
    }
]


# 5-2. 도구(함수) 목록 정의 (OpenAI tool-call 형식)
eval_tools = [
    {
        "type": "function",
        "function": {
            "name": "cal_evalscore",
            "description": "[사용자 아이디어]와 [특허 문서 조각]의 유사도를 분석하여, 0-100점 사이의 점수와 그 근거를 반환합니다.",
            "parameters": {
                "type": "object",
                "properties": {
                    "eval_score": {
                        "type": "integer",
                        "description": "[사용자 아이디어]와 [특허 문서 조각] 간의 기술적 유사도 점수. 0 (완전히 무관함)에서 100 (기술적으로 동일함) 사이의 정수입니다.",
                        "minimum": 0,
                        "maximum": 100
                    },
                    "reason": {
                        "type": "string",
                        "description": "해당 점수를 부여한 구체적인 이유. 특허 조각의 어느 부분이 아이디어의 어떤 개념과 유사한지(또는 다른지) 명확히 짚어서 설명해야 합니다."
                    }
                },
                "required": ["eval_score", "reason"],
            },
        },
    }
]

TOOL_MAPPING = {"search_query": search_query}

def get_unique_patents(results):
    """
    ChromaDB 검색 결과에서 청구 번호(ApplicationNumber) 기준으로 중복을 제거하고,
    각 특허별 가장 유사도가 높은(distance가 낮은) 청크만 남겨 상위 k개를 반환합니다.
    """
    
    # 1. 중복 제거를 위한 딕셔너리 (Key: 청구번호, Value: 해당 특허의 베스트 청크 정보)
    unique_patents = {}
    
    # 검색된 결과의 개수만큼 반복
    num_results = len(results['documents'][0])
    
    for i in range(num_results):
        # 정보 추출
        metadata = results['metadatas'][0][i]
        document = results['documents'][0][i]
        distance = results['distances'][0][i] # 코사인 유사도 거리 (낮을수록 유사함)
        
        # 청구 번호 추출 (그룹화의 기준 Key)
        app_number = metadata.get('ApplicationNumber')
        
        # 예외 처리: 청구 번호가 없는 경우 스킵 (데이터 무결성 체크)
        if not app_number:
            continue
            
        # 2. 그룹화 및 최적 청크 선별 로직
        if app_number not in unique_patents:
            # (A) 처음 발견된 특허라면 -> 딕셔너리에 저장
            unique_patents[app_number] = {
                "metadata": metadata,
                "document": document,
                "distance": distance
            }
        else:
            # (B) 이미 발견된 특허라면 -> 더 유사한지(distance가 더 작은지) 비교
            existing_distance = unique_patents[app_number]['distance']
            
            if distance < existing_distance:
                # 현재 청크가 기존 청크보다 더 유사하다면 정보 갱신
                unique_patents[app_number] = {
                    "metadata": metadata,
                    "document": document,
                    "distance": distance
                }
    
    # 3. 딕셔너리를 리스트로 변환
    unique_list = list(unique_patents.values())
    
    # 4. 거리(distance) 기준으로 오름차순 정렬 (낮은게 1등)
    unique_list.sort(key=lambda x: x['distance'])
    
    # 5. 사용자가 원하는 개수(target_k)만큼 자르기
    #final_results = unique_list[:target_k]
    final_results = unique_list
    
    return final_results

def search_query(query, db_path="./patent_chroma_db", collection_name="patents", model_name="nlpai-lab/KURE-v1", n_results=20):
    """
    지정된 ChromaDB에서 아이디어(쿼리 텍스트)를 검색합니다.
    """
    query_text = query
    print(f"\n--- 테스트 검색 시작 ---")
    print(f"Query: '{query_text}'")
    
    try:
        # 1. DB 클라이언트 초기화
        client = chromadb.PersistentClient(path=db_path)
        
        # 2. 임베딩 함수 설정 (DB에 저장할 때 사용한 것과 동일해야 함)
        try:
            embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name=model_name
            )
        except Exception as e:
            print(f"검색을 위한 임베딩 모델 로드 중 오류 발생: {e}")
            return
        
        # 3. 컬렉션 가져오기 (get_collection 사용)
        try:
            collection = client.get_collection(
                name=collection_name,
                embedding_function=embedding_func
            )
            print(f"'{collection_name}' 컬렉션 (문서 {collection.count()}개)을 성공적으로 불러왔습니다.")
        except Exception as e:
            print(f"'{collection_name}' 컬렉션 가져오기 중 오류 발생: {e}")
            print("'process_patents_to_chroma' 함수가 먼저 성공적으로 실행되었는지 확인하세요.")
            return

        # 4. 쿼리 실행
        results = collection.query(
            query_texts=[query_text],
            n_results=n_results,
            include=["metadatas", "documents", "distances"] # 거리(유사도)도 포함
        )
        
        print(f"\n--- 검색 결과 (상위 {len(results.get('ids', [[]])[0])}개) ---")
        
        # 5. 결과 출력
        if not results or not results.get('ids', [[]])[0]:
            print("검색 결과가 없습니다.")
            return
        results = get_unique_patents(results) #중복 특허 제거

        return results
            
    except Exception as e:
        print(f"검색 중 예상치 못한 오류가 발생했습니다: {e}")
        
def evaluation_idea(user_idea: str, patent_chunk: str, model_name: str = "x-ai/grok-4.1-fast"):
    print(f"[사용자 아이디어]: {user_idea}")
    print(f"[특허 문서 조각]: {patent_chunk[:100]}")

    user_query = f"[사용자 아이디어]: {user_idea}\n\n[특허 문서 조각]: {patent_chunk}"
    
    messages = [
  {
    "role": "system",
    "content": evaluation_system_prompt
  },
  {
    "role": "user",
    "content": user_query,
  }
]
    request = {
    "model": model_name,
    "tools": eval_tools,
    "messages": messages
}
    try:
        # 1. '아이디어 게이트키퍼' LLM 호출
        response = openai_client.chat.completions.create(**request)


        print("\n[게이트키퍼 LLM 응답]")
        print("--------------------")
        print("--------------------")
        print(response.choices[0].message.content)
        
        for tool_call in response.choices[0].message.tool_calls:

            tool_name = tool_call.function.name
            tool_args = json.loads(tool_call.function.arguments)

        eval_result = [tool_args["eval_score"], tool_args["reason"]]
        return eval_result
        

    except Exception as e:
        print(f"\n--- [오류] LLM API 호출 또는 라우팅 중 오류 발생 ---")
        print(f"에러 상세: {e}")
        return {"status": "error", "message": str(e)}
        

OPENROUTER_API_KEY = ""

openai_client = OpenAI(
  base_url="https://openrouter.ai/api/v1",
  api_key=OPENROUTER_API_KEY,
)

def execute_router(user_query: str, model_name: str = "x-ai/grok-4.1-fast"):
    """
    사용자 아이디어를 받아 게이트키퍼 LLM을 호출하고,
    결과에 따라 RAG 검색을 트리거하거나 사용자에게 피드백을 반환합니다.
    """

    print(f"\n--- [EXECUTE ROUTER] ---")
    print(f"입력 아이디어: '{user_query}'")

    messages = [
  {
    "role": "system",
    "content": router_system_prompt
  },
  {
    "role": "user",
    "content": user_query,
  }
]
    request = {
    "model": model_name,
    "tools": search_tools,
    "messages": messages
}
    try:
        # 1. '아이디어 게이트키퍼' LLM 호출
        response = openai_client.chat.completions.create(**request)


        print("\n[게이트키퍼 LLM 응답]")
        print("--------------------")
        print("--------------------")
        print(response.choices[0].message.content)

        for tool_call in response.choices[0].message.tool_calls:

            tool_name = tool_call.function.name
            tool_args = json.loads(tool_call.function.arguments)
            improved_query = tool_args['query']

        eval_results = []
        patent_chunks = TOOL_MAPPING[tool_name](**tool_args)
        for idx, item in enumerate(patent_chunks):
            document = item['document']
            eval_result = evaluation_idea(improved_query, document)
            eval_results.append(eval_result)

        print(eval_results)
    except Exception as e:
        print(f"\n--- [오류] LLM API 호출 또는 라우팅 중 오류 발생 ---")
        print(f"에러 상세: {e}")
        return {"status": "error", "message": str(e)}
        

