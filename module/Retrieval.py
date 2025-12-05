import pandas as pd
import chromadb
import os
import csv
from chromadb.utils import embedding_functions

GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")

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

def search_query(query_text, db_path="./patent_chroma_db", collection_name="patents", model_name="gemini-embedding-001", n_results=20):
    """
    지정된 ChromaDB에서 아이디어(쿼리 텍스트)를 검색합니다.
    """
    print(f"\n--- 테스트 검색 시작 ---")
    print(f"Query: '{query_text}'")
    
    try:
        # 1. DB 클라이언트 초기화
        client = chromadb.PersistentClient(path=db_path)
        
        # 2. 임베딩 함수 설정 (DB에 저장할 때 사용한 것과 동일해야 함)
        try:
            embedding_func = embedding_functions.GoogleGenerativeAiEmbeddingFunction(
    api_key=GOOGLE_API_KEY,
    model_name=model_name # 최신 모델 권장
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
