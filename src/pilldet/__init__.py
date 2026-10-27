"""pilldet – 경구약제 이미지 객체 검출 프로젝트의 공통 코드.

모듈 구성
- config   : 경로 탐색과 공통 설정
- data     : 라벨 로딩, 이미지 읽기, 파일명 파싱
- boxes    : 박스 좌표 변환과 IoU 계산
- segment  : 배경 차분 기반 알약 영역 분할
- clean    : 라벨 감사와 규칙 기반 수정
- split    : 조합 단위 Stratified Group K-Fold
- export   : YOLO / COCO 내보내기와 왕복 검증
- viz      : 시각화 공통 함수
"""

__version__ = "0.1.0"
