import jwt
import datetime

API_KEY = "devkey"
API_SECRET = "secret"

def create_token(identity: str, room_name: str) -> str:
    """
    LiveKit용 JWT 토큰 생성 (VideoGrants 직접 구성)

    Args:
        identity (str): 참가자 ID (예: "ksl_worker")
        room_name (str): 입장할 방 이름 (예: "dev-room")

    Returns:
        str: 서명된 JWT 토큰
    """

    now = datetime.datetime.utcnow()

    # LiveKit 권한(비디오 그랜트) 설정
    video_grants = {
        "roomCreate": True,       # 방 만들기 권한
        "roomJoin": True,         # 방 입장 권한
        "room": room_name,        # 입장 가능한 방 이름
        "canPublish": False,      # 미디어 publish 가능 여부
        "canSubscribe": True,     # 구독 가능 여부
        "canPublishData": True,   # 데이터 채널 publish 가능 여부
    }

    # LiveKit이 기대하는 payload 형식
    payload = {
        "iss": API_KEY,          # 발급자: LiveKit API Key
        "sub": identity,         # 토큰 주체: 참가자 identity
        "iat": now,              # 발급 시간
        "exp": now + datetime.timedelta(hours=1),  # 만료 시간 (1시간 유효)
        "video": video_grants,   # 비디오 권한
    }

    # HS256 알고리즘으로 서명
    token = jwt.encode(
        payload,
        API_SECRET,
        algorithm="HS256",
        headers={"typ": "JWT", "alg": "HS256"}
    )

    # PyJWT 2.x는 str을 반환하지만, 혹시 bytes면 decode
    if isinstance(token, bytes):
        token = token.decode("utf-8")

    return token