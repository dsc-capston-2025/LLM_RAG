from flask import Flask, request, jsonify, views
from flask_cors import CORS
from openai import OpenAI
from module.Generator import execute_router
from dotenv import load_dotenv
import os
import traceback

GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")

api_client = OpenAI(
  base_url='https://generativelanguage.googleapis.com/v1beta/openai/',
  api_key=GOOGLE_API_KEY,
)


app = Flask(__name__)

# 개발 중인 React 앱(예: localhost:3000)에서 Flask 서버(예: localhost:5000)로 API를 요청할 때 필요.
CORS(app)


def process_idea_text(text: str) -> dict:
    """
    Frontend에서 받은 텍스트(아이디어)를 처리하는 로직

   인터페이스 정의를 위한 임시 데이터 정의 
    """
    success_bool, eval_results, abstract_result = execute_router(text, model_name="gemini-2.5-flash", api_client=api_client)
    patent_list = []
    print(f"success_bool: {success_bool} 확인")
    if  success_bool:
        for i in range(len(eval_results)):
            patent_data = {
                "matchstatus": "success",
                "patentId": eval_results[i][0].get('ApplicationNumber'),
                "title": eval_results[i][0].get('InventionName'),
                "applicationDate": eval_results[i][0].get('ApplicationDate'),
                "applicant": eval_results[i][0].get('Applicant'),
                "summary": eval_results[i][1][1],
                "relevanceScore": str(int(eval_results[i][1][0])/100)
                    }
            patent_list.append(patent_data)

    response = {
            "status": "success" if success_bool else "failed",
            "chatResponse": abstract_result,
            "patentList": patent_list
            }
    
    return response


@app.route('/api/analyze-idea', methods=['POST'])
def analyze_idea():
    """
    Frontend로부터 아이디어 텍스트를 받아 처리하고 결과를 반환하는 API 엔드포인트
    """
    try:
        # Frontend에서 보낸 JSON 데이터를 수신.
        # React에서 '{"idea_text": "..."}'와 같은 형태로 보낸다고 가정.
        data = request.json
        idea_text = data.get('idea')

        if not idea_text:
            return jsonify({"status": "error", "message": "No 'idea_text' provided."}), 400

        # 텍스트를 처리.
        response_data = process_idea_text(idea_text)

        # 처리된 결과를 JSON 형태로 Frontend에 반환.
        return jsonify(response_data)

    except Exception as e:
        error_trace = traceback.format_exc()
        print(error_trace)
        print(f"Error occurred: {e}")
        return jsonify({"status": "error", "message": "An internal server error occurred."}), 500

if __name__ == '__main__':
    # debug=True: 개발 중에 코드가 변경되면 서버를 자동으로 재시작.
    app.run(debug=True, port=5000)
