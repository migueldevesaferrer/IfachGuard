import os
import uuid

import streamlit as st
from langchain_core.messages import HumanMessage

from ifachguard_agent import configuracion_thread, contenido_a_texto, crear_agente


@st.cache_resource(show_spinner=False)
def obtener_agente(api_key: str):
    return crear_agente(api_key)


def obtener_api_key() -> str | None:
    api_key = os.getenv("GEMINI_API_KEY")
    if api_key:
        return api_key
    try:
        return st.secrets["GEMINI_API_KEY"]
    except Exception:
        return None


st.set_page_config(page_title="IfachGuard", page_icon="💬")
st.title("💬 IfachGuard")
st.write(
    "Asistente del Parque Natural del Penyal d'Ifac. Pregunta sobre normativa, "
    "uso público, conservación, flora, fauna o escalada."
)

api_key = obtener_api_key()
if not api_key:
    st.info(
        "Configura GEMINI_API_KEY en `.streamlit/secrets.toml` para desarrollo local "
        "o en Secrets de Streamlit Community Cloud."
    )
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Hola. ¿Qué quieres saber sobre el Parque Natural del Penyal d'Ifac?",
        }
    ]
if "thread_id" not in st.session_state:
    st.session_state.thread_id = uuid.uuid4().hex

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Escribe tu consulta sobre el parque..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Consultando el corpus del parque..."):
            try:
                agente = obtener_agente(api_key)
                resultado = agente.invoke(
                    {"messages": [HumanMessage(content=prompt)]},
                    config=configuracion_thread(st.session_state.thread_id),
                )
                response = contenido_a_texto(resultado["messages"][-1].content)
            except Exception as error:
                st.error(f"No se pudo completar la consulta: {error}")
            else:
                st.markdown(response)
                st.session_state.messages.append(
                    {"role": "assistant", "content": response}
                )
