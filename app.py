from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)

# 개발 중인 React 앱(예: localhost:3000)에서 Flask 서버(예: localhost:5000)로 API를 요청할 때 필요.
CORS(app)
------------------------------------------------------------------
def process_idea_text(text: str) -> dict:
    """
    Frontend에서 받은 텍스트(아이디어)를 처리하는 로직

   인터페이스 정의를 위한 임시 데이터 정의 
    """
    print(f"전달받은 아이디어 텍스트: {text}")

    #response = my_patent_analysis_module.analyze(text)

    success_response = {
        "status": "success",
        "chatResponse": "귀하의 아이디어와 유사도가 높은 3건의 특허를 찾았습니다. 핵심 키워드는 'A'와 'B'이며, 전반적인 기술 동향은 다음과 같습니다...",
        "patentList": [
            {
                "matchstatus": "failed",
                "patentId": "KR1020230012345",
                "title": "첫 번째 유사 특허 제목",
                "applicationDate": "2023-01-10",
                "applicant": "특허 출원인 A",
                "summary": "별로 유사하지 않은 특허 같아요",
                "relevanceScore": 0.32
            },
            {
                "matchstatus": "success",
                "patentId": "US20220056789A1",
                "title": "두 번째 유사 특허 제목",
                "applicationDate": "2022-05-20",
                "applicant": "특허 출원인 B",
                "summary": "B 기술을 C 분야에 적용하는 혁신적인 방안을 제시합니다...",
                "relevanceScore": 0.88
            },
            {
                "matchstatus": "success",
                "patentId": "JP20210098765",
                "title": "세 번째 유사 특허 제목",
                "applicationDate": "2021-11-30",
                "applicant": "특허 출원인 C",
                "summary": "기존 A 기술의 단점을 보완하는 새로운 아키텍처를 제안합니다...",
                "relevanceScore": 0.85
            }
        ]
    }

  failed_response = {
"status": "clarification_needed",
"chatResponse": "아이디어를 분석했지만, 내용이 너무 광범위합니다. 'AI'라고 하셨는데, '컴퓨터 비전' 분야인가요, '자연어 처리' 분야인가요? 조금 더 구체적으로 알려주세요.",
"patentList": []
}
    
    return success_response


@app.route('/api/analyze-idea', methods=['POST'])
def analyze_idea():
    """
    Frontend로부터 아이디어 텍스트를 받아 처리하고 결과를 반환하는 API 엔드포인트
    """
    try:
        # Frontend에서 보낸 JSON 데이터를 수신.
        # React에서 '{"idea_text": "..."}'와 같은 형태로 보낸다고 가정.
        data = request.json
        idea_text = data.get('idea_text')

        if not idea_text:
            return jsonify({"status": "error", "message": "No 'idea_text' provided."}), 400

        # 텍스트를 처리.
        response_data = process_idea_text(idea_text)

        # 처리된 결과를 JSON 형태로 Frontend에 반환.
        return jsonify(response_data)

    except Exception as e:
        print(f"Error occurred: {e}")
        return jsonify({"status": "error", "message": "An internal server error occurred."}), 500

if __name__ == '__main__':
    # debug=True: 개발 중에 코드가 변경되면 서버를 자동으로 재시작.
    app.run(debug=True, port=5000)
