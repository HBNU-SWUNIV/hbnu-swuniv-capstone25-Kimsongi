from transformers import pipeline

# KoAlpaca 모델 로드
generator = pipeline(
    "text-generation",
    model="beomi/KoAlpaca-Polyglot-12.8B",
    device_map="auto"
)

def generate_sentence_llm(words):
    """
    단어 리스트를 받아 LLM을 통해 자연스러운 문장 생성
    """
    prompt = f"다음 단어들을 사용해서 자연스러운 영어 문장을 만들어줘:\n단어: {', '.join(words)}\n문장:"
    result = generator(prompt, max_length=50, do_sample=True, temperature=0.7)
    return result[0]['generated_text']

# 테스트
if __name__ == "__main__":
    words = []
    sentence = generate_sentence_llm(words)
    print("📝 생성된 문장:", sentence)
