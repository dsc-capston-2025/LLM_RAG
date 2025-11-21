from Retrieval import search_query
import json, requests
from openai import OpenAI

## 라우터 llm 시스템 프롬프트
ROUTER_SYSTEM_PROMPT = """
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
EVALUATION_SYSTEM_PROMPT = """
당신은 사용자의 아이디어와 유사한 선행기술을 찾아내는 '선행기술 조사(Prior Art Search) 전문가'입니다.

[특허 문서 조각]은 1차 AI 검색(RAG)을 통해 [사용자 아이디어]와 유사할 가능성이 있어 검색된 결과입니다.

당신의 핵심 임무는 **[사용자 아이디어]와 [특허 문서 조각] 사이의 '기술적 교집합(Similarity)'을 명확히 식별**하는 것입니다. 두 내용이 완벽히 똑같지 않더라도, 아이디어의 핵심 구성요소나 해결 원리가 특허에 포함되어 있다면 그 유사성을 적극적으로 찾아내어 설명해야 합니다.

당신의 응답은 반드시 `cal_evalscore` 함수를 호출하는 것이어야 합니다.

---
### ⚖️ 평가 및 분석 지침

1.  **`eval_score` (0-100% 사이의 정수):**
    * 이 점수는 [사용자 아이디어]가 [특허 문서 조각]에 의해 **'기술적으로 얼마나 커버되는가(유사도)'**를 나타냅니다.
    * 비판보다는 **'연관성 발견'**에 초점을 맞추어 점수를 부여하세요.

    * **0~24% (낮은 연관성):** 단순 키워드만 겹칠 뿐, 기술적 해결 원리가 전혀 다릅니다.
    * **25~49% (부분 유사):** 기술 분야나 적용 대상은 다르지만, **'기반이 되는 기술적 메커니즘'**이나 **'아이디어의 일부 구성요소'**가 유사합니다. (예: '드론 배송' 아이디어 vs '로봇 배송' 특허)
    * **50~74% (높은 유사성):** 해결하려는 문제와 목적이 같고, 핵심적인 기술 수단이 상당 부분 겹칩니다. (강력한 선행기술 후보)
    * **75~100% (실질적 동일):** [사용자 아이디어]의 핵심 발명이 [특허 문서 조각]에 이미 구체적으로 구현되어 있습니다.

2.  **`reason` (문자열):**
    * **[핵심 요구사항]** 차이점을 설명하는 것도 중요하지만 **'어떤 부분이 유사한지'**를 중점적으로 설명하세요.
    * **작성 구조:**
        1.  **[유사성 분석]:** "[특허]의 A기술은 [아이디어]의 B개념과 기술적으로 유사합니다."와 같이 **구체적인 매칭 포인트**를 먼저 서술합니다.
        2.  **[차이점/한계]:** 그 후, 유사함에도 불구하고 점수가 어떤 부분이 차이가 나는지(분야의 차이, 구체적 구현 방식의 차이 등)를 덧붙여 균형을 맞춥니다.
    * **예시:** "이 특허는 [아이디어]와 마찬가지로 'RAG를 활용한 검색 보정' 방식을 사용한다는 점에서 핵심 원리가 일치합니다. 다만, 적용 분야가 [아이디어]는 '특허'인 반면 이 문서는 '일반 웹 검색'이라는 점에서 차이가 있어 60점을 부여합니다."

---

이제 [사용자 아이디어]와 [특허 문서 조각]을 비교 분석하여, **유사성을 중심으로** 평가하고 `cal_evalscore` 함수를 호출하세요.
"""

ABSTRACTOR_SYSTEM_PROMPT = """
# Role Definition
당신은 숙련된 특허 변리사이자 R&D 기술 컨설턴트입니다. 당신의 임무는 사용자의 아이디어와 이를 기반으로 검색된 '유사 특허 리스트'를 종합적으로 분석하여, 사용자에게 통찰력 있는 최종 보고서를 제공하는 것입니다.

# Input Data Description
당신에게는 다음과 같은 정보가 텍스트 형식으로 제공됩니다:
1. [사용자 아이디어]: 사용자가 입력한 발명 아이디어
2. [검색된 특허]: 검색된 특허들의 리스트. 각 항목은 다음을 포함함:
   - 특허 제목
   - LLM Judge가 분석한 주요 유사점 및 차이점

# Task Objectives
제공된 정보를 바탕으로 다음 섹션을 포함하는 마크다운(Markdown) 형식의 보고서를 작성하세요.

## 1. 종합 검토 의견 (Executive Summary)
- 사용자의 아이디어가 기존 선행 기술들과 비교했을 때 등록 가능성이 있는지, 혹은 기술적 장벽이 높은지 전반적인 난이도를 3~4문장으로 요약하세요.
- 가장 유사도가 높은 특허 1~2개를 특정하여 언급하고, 핵심적인 겹치는 기술 요소를 지적하세요.

## 2. 기술적 제언 (Strategic Advice)
- 사용자가 아이디어를 구체화하거나 특허를 출원하기 위해 보완해야 할 기술적 공백(White Space)이나 회피 설계 방향을 제시하세요.

# Guidelines & Constraints
- **Hallucination 방지:** 제공된 [Retrieved Patents]에 없는 내용은 절대 지어내지 마세요.
- **객관성 유지:** 사용자의 아이디어를 무조건 칭찬하기보다, 냉철하게 선행 기술과의 중복성을 지적하는 것이 사용자에게 더 도움이 됩니다.
- **가독성:** 전문 용어를 사용하되, 학부생 수준의 엔지니어가 이해할 수 있도록 명확하게 서술하세요.
- **언어:** 한국어(Korean)로 작성하세요.

# Tone
- 전문적이고, 분석적이며, 객관적인 톤을 유지하세요.
"""

# --- 5. Function Calling 테스트 ---
SEARCH_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_query",
            "description": "사용자의 아이디어가 구체적일 때, 관련 특허 문서를 검색하기 위해 호출합니다. RAG 시스템을 통해 의미적으로 유사한 특허를 반환합니다.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query_text": {
                        "type": "string",
                        "description": "RAG 검색을 위한 기술적 서술문입니다. 단순 키워드 나열(예: 'A B C')을 절대 금지합니다. 대신 'A의 기능을 수행하기 위해 B에 결합된 C 장치'와 같이 구성 요소 간의 관계와 목적이 명확한 문장 형태(특허 명칭 스타일)로 입력해야 합니다."
                    }
                },
                "required": ["query_text"],
            },
        },
    }
]


# 5-2. 도구(함수) 목록 정의 (OpenAI tool-call 형식)
EVAL_TOOLS = [
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

        
def evaluation_idea(user_idea: str, patent_chunk: str, model_name: str = "x-ai/grok-4.1-fast", api_client=api_client):
    print(f"[사용자 아이디어]: {user_idea}")
    print(f"[특허 문서 조각]: {patent_chunk[:100]}")

    user_query = f"[사용자 아이디어]: {user_idea}\n\n[특허 문서 조각]: {patent_chunk}"
    
    messages = [
  {
    "role": "system",
    "content": EVALUATION_SYSTEM_PROMPT
  },
  {
    "role": "user",
    "content": user_query,
  }
]
    request = {
    "model": model_name,
    "tools": EVAL_TOOLS,
    "messages": messages
}
    try:
        response = api_client.chat.completions.create(**request)

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

def abstract_result(user_query, eval_results, api_client=api_client):
    result = []
    details = "[사용자 아이디어]\n-{user_query}\n\n[검색된 특허]\n"
    
    for i in eval_details:
        details += f"{i+1}.\n-제목: {eval_results[i][0].get('InventionName')}\n-평가정보: {eval_results[i][1][1]}\n\n"
    
    messages = [
  {
    "role": "system",
    "content": ABSTRACTOR_SYSTEM_PROMPT
  },
  {
    "role": "user",
    "content": details,
  }
]
    request = {
    "model": model_name,
    "messages": messages
}

    try:
        # 1. '아이디어 게이트키퍼' LLM 호출
        response = api_client.chat.completions.create(**request)

        response_content = response.choices[0].message.content
        print("\n[게이트키퍼 LLM 응답]")
        print("--------------------")
        print("--------------------")
        print(response_content)
        
        return response_content
            
    except Exception as e:
        print(f"\n--- [오류] LLM API 호출 또는 라우팅 중 오류 발생 ---")
        print(f"에러 상세: {e}")

def execute_router(user_query: str, model_name: str = "x-ai/grok-4.1-fast", api_client=api_client):
    success_bool = False
    """
    사용자 아이디어를 받아 게이트키퍼 LLM을 호출하고,
    결과에 따라 RAG 검색을 트리거하거나 사용자에게 피드백을 반환합니다.
    """

    print(f"\n--- [EXECUTE ROUTER] ---")
    print(f"입력 아이디어: '{user_query}'")

    messages = [
  {
    "role": "system",
    "content": ROUTER_SYSTEM_PROMPT
  },
  {
    "role": "user",
    "content": user_query,
  }
]
    request = {
    "model": model_name,
    "tools": SEARCH_TOOLS,
    "messages": messages
}
    try:
        # 1. '아이디어 게이트키퍼' LLM 호출
        response = api_client.chat.completions.create(**request)
        response_text = response.choices[0].message.content

        print("\n[게이트키퍼 LLM 응답]")
        print("--------------------")
        print("--------------------")
        print(response.choices[0].message.content)
        
        if not response.choices[0].message.tool_calls:
            return success_bool, None, response_text

        success_bool = True
        
        for tool_call in response.choices[0].message.tool_calls:

            tool_name = tool_call.function.name
            tool_args = json.loads(tool_call.function.arguments)
            improved_query = tool_args['query']

        eval_results = []
        patent_chunks = TOOL_MAPPING[tool_name](**tool_args)
        for idx, item in enumerate(patent_chunks):
            patent_metadata = item['metadata']
            document = item['document']
            eval_result = evaluation_idea(improved_query, document)
            eval_results.append((patent_metadata, eval_result))

        result = abstract_result(user_query, eval_results, api_client)
        
        return success_bool, eval_results, result
    except Exception as e:
        print(f"\n--- [오류] LLM API 호출 또는 라우팅 중 오류 발생 ---")
        print(f"에러 상세: {e}")
        return "error", None, str(e)
        

