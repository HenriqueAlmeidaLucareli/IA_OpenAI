from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    HumanMessagePromptTemplate,
    AIMessagePromptTemplate,
    SystemMessagePromptTemplate,
    FewShotChatMessagePromptTemplate
)
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI
from datetime import datetime, timezone, timedelta
from dotenv import load_dotenv
import unicodedata
import sys
import io
import os

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stdin = io.TextIOWrapper(sys.stdin.buffer, encoding='utf-8')

load_dotenv()

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.7,
    api_key=os.getenv("OPENAI_API_KEY")
)

fuso_brasilia = timezone(timedelta(hours=-3))
today = datetime.now(fuso_brasilia)

system_prompt = SystemMessagePromptTemplate.from_template("""
🏋️‍♂️ Nome da empresa:
Vitality Fitness Center

🏢 Descrição da empresa:
A Vitality Fitness Center é uma academia completa focada em oferecer não apenas musculação de alta qualidade, mas também uma variedade de modalidades para atender todos os perfis de alunos.
Nossa missão é promover saúde, bem-estar e qualidade de vida através de treinos personalizados e acompanhamento de profissionais qualificados.

🎯 Missão:
Transformar vidas por meio do movimento, promovendo saúde física, mental e social.

👓 Visão:
Ser referência em inovação e excelência no segmento fitness e de bem-estar na nossa região até 2030.

💎 Valores:
Comprometimento com resultados

Respeito às individualidades

Inovação constante

Ambiente acolhedor e motivador

Trabalho em equipe

📋 Modalidades oferecidas:
Musculação

Pilates (solo e aparelhos)

Treinamento funcional

Jiu-Jitsu (adulto e infantil)

Cross training

Yoga

Alongamento

Zumba e ritmos

Aulas de HIIT (High-Intensity Interval Training)

👥 Público-alvo:
Jovens e adultos entre 18 e 50 anos

Idosos que buscam melhor qualidade de vida

Crianças e adolescentes para aulas de artes marciais e funcional kids

Pessoas em reabilitação ou buscando treino de baixo impacto (Pilates, Yoga)

🛠️ Estrutura física:
Área de musculação com equipamentos modernos

Estúdios climatizados para aulas de grupo

Espaço de lutas com tatame profissional

Sala de Pilates com equipamentos específicos

Área de descanso e convivência

Vestiários completos

Estacionamento próprio

📍 Localização:
Rua das Palmeiras, 123 – Bairro Saúde Viva – São Paulo, SP

📞 Contato:
Telefone: (11) 99999-1234

WhatsApp: (11) 98888-5678

E-mail: contato@vitalityfitness.com.br

Instagram: @vitalityfitness.br

Site: www.vitalityfitness.com.br

🕑 Horários de funcionamento:
Segunda a sexta: 6h às 22h

Sábado: 8h às 16h

Domingo: fechado
""")

example_prompt = ChatPromptTemplate.from_messages([
    HumanMessagePromptTemplate.from_template("{human}"),
    AIMessagePromptTemplate.from_template("{ai}")
])

shots = [
    {
        "human": "A academia abre de Domingo?",
        "ai": "- Olá a academia não abre de domingo\n- Recomendação:\nA abre de segunda a sexta, das 6h às 22h, e aos sábados, das 8h às 16h."
    }
]

fewshots = FewShotChatMessagePromptTemplate(
    examples=shots,
    example_prompt=example_prompt
)

prompt = ChatPromptTemplate.from_messages([
    system_prompt,
    fewshots,
    MessagesPlaceholder(variable_name="chat_history"),
    HumanMessagePromptTemplate.from_template("{input}")
])

prompt = prompt.partial(today_local=today.isoformat())

#  Histórico
store = {}

def get_session_history(session_id: str):
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

# Chain direto (sem agent, por enquanto)
chain = prompt | llm

chain_with_history = RunnableWithMessageHistory(
    chain,
    get_session_history=get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history"
)

def sanitize_input(text: str) -> str:
    return ''.join(
        c for c in unicodedata.normalize('NFD', text)
        if unicodedata.category(c) != 'Mn'
    )

# Main
while True:
    try:
        user_input = input("Você: ")

        if user_input.lower() == "sair":
            print("Encerrando...")
            break

        safe_input = sanitize_input(user_input)

        resposta = chain_with_history.invoke(
            {"input": safe_input},
            config={"configurable": {"session_id": "user1"}}
        )

        print("Bot:", resposta.content)

    except Exception as e:
        print("Erro:", e)