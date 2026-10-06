import streamlit as st
import chromadb
import ollama

from pypdf import PdfReader
from sentence_transformers import SentenceTransformer


# ============================================================
# STREAMLIT CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="PDF RAG Assistant",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 PDF RAG Assistant")
st.caption(
    "Upload a PDF and ask questions based on its content."
)


# ============================================================
# LOAD EMBEDDING MODEL
# ============================================================

@st.cache_resource
def get_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


embedding_model = get_embedding_model()


# ============================================================
# INITIALIZE CHROMADB
# ============================================================

@st.cache_resource
def get_chroma_collection():

    client = chromadb.PersistentClient(
        path="./chroma_db"
    )

    collection = client.get_or_create_collection(
        name="pdf_rag_collection"
    )

    return collection


collection = get_chroma_collection()


# ============================================================
# EXTRACT TEXT FROM PDF
# ============================================================

def extract_pdf_text(uploaded_file):

    reader = PdfReader(uploaded_file)

    pages_text = []

    for page in reader.pages:

        text = page.extract_text()

        if text:
            pages_text.append(text)

    return "\n".join(pages_text)


# ============================================================
# SPLIT TEXT INTO CHUNKS
# ============================================================

def create_chunks(
    text,
    chunk_size=1000,
    chunk_overlap=150
):

    text = text.strip()

    if not text:
        return []

    chunks = []

    start = 0
    text_length = len(text)

    while start < text_length:

        end = start + chunk_size

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end >= text_length:
            break

        start = end - chunk_overlap

    return chunks


# ============================================================
# STORE DOCUMENTS IN CHROMADB
# ============================================================

def store_chunks(chunks):

    # Create embeddings
    embeddings = embedding_model.encode(
        chunks,
        convert_to_numpy=True
    )

    # Unique IDs
    ids = [
        f"chunk_{i}"
        for i in range(len(chunks))
    ]

    # Add to ChromaDB
    collection.add(
        ids=ids,
        documents=chunks,
        embeddings=embeddings.tolist()
    )


# ============================================================
# SEARCH RELEVANT DOCUMENT CHUNKS
# ============================================================

def search_documents(question, number_of_results=5):

    # Convert question into embedding
    question_embedding = embedding_model.encode(
        question,
        convert_to_numpy=True
    )

    # Search ChromaDB
    results = collection.query(
        query_embeddings=[
            question_embedding.tolist()
        ],
        n_results=number_of_results
    )

    documents = results.get("documents", [])

    if not documents:
        return []

    return documents[0]


# ============================================================
# GENERATE ANSWER USING OLLAMA
# ============================================================

def generate_answer(question, context):

    prompt = f"""
You are a helpful document question-answering assistant.

Answer the user's question using ONLY the information
provided in the context.

Rules:
1. Do not make up information.
2. Do not use outside knowledge.
3. If the answer is not present in the context,
   say that the information was not found in the document.
4. Give a clear and concise answer.

Context:
--------------------------------------------------
{context}
--------------------------------------------------

User Question:
{question}

Answer:
"""

    response = ollama.chat(
        model="llama3.2",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return response["message"]["content"]


# ============================================================
# CLEAR DATABASE
# ============================================================

def clear_database():

    global collection

    try:
        client = chromadb.PersistentClient(
            path="./chroma_db"
        )

        try:
            client.delete_collection(
                "pdf_rag_collection"
            )
        except Exception:
            pass

        collection = client.get_or_create_collection(
            name="pdf_rag_collection"
        )

        return True

    except Exception:
        return False


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("⚙️ Settings")

    st.write(
        "**Embedding Model:**"
    )

    st.code(
        "all-MiniLM-L6-v2"
    )

    st.write(
        "**Vector Database:**"
    )

    st.code(
        "ChromaDB"
    )

    st.write(
        "**LLM:**"
    )

    st.code(
        "llama3.2"
    )

    st.divider()

    if st.button(
        "🗑️ Clear Database",
        use_container_width=True
    ):

        if clear_database():

            st.session_state["pdf_processed"] = False

            st.success(
                "Database cleared."
            )

            st.rerun()

        else:

            st.error(
                "Could not clear database."
            )


# ============================================================
# PDF UPLOAD
# ============================================================

st.header("📄 Upload PDF")

uploaded_file = st.file_uploader(
    "Choose a PDF file",
    type=["pdf"]
)


# ============================================================
# PROCESS PDF
# ============================================================

if uploaded_file is not None:

    st.info(
        f"Selected file: {uploaded_file.name}"
    )

    if st.button(
        "🔄 Process PDF",
        type="primary"
    ):

        with st.spinner(
            "Extracting text from PDF..."
        ):

            try:

                pdf_text = extract_pdf_text(
                    uploaded_file
                )

            except Exception as error:

                st.error(
                    "Error while reading PDF."
                )

                st.exception(error)

                st.stop()

        if not pdf_text.strip():

            st.error(
                "No readable text was found in this PDF."
            )

        else:

            chunks = create_chunks(
                pdf_text
            )

            if not chunks:

                st.error(
                    "Could not create text chunks."
                )

            else:

                with st.spinner(
                    "Creating embeddings and storing document..."
                ):

                    try:

                        # Clear old collection first
                        clear_database()

                        # Store new PDF
                        store_chunks(chunks)

                    except Exception as error:

                        st.error(
                            "Error while storing document."
                        )

                        st.exception(error)

                        st.stop()

                st.session_state[
                    "pdf_processed"
                ] = True

                st.session_state[
                    "file_name"
                ] = uploaded_file.name

                st.session_state[
                    "chunk_count"
                ] = len(chunks)

                st.success(
                    f"PDF processed successfully! "
                    f"{len(chunks)} chunks created."
                )


# ============================================================
# DOCUMENT STATUS
# ============================================================

if st.session_state.get(
    "pdf_processed",
    False
):

    st.success(
        f"📚 Document ready: "
        f"{st.session_state.get('file_name', 'PDF')}"
    )

    st.write(
        f"Number of chunks: "
        f"**{st.session_state.get('chunk_count', 0)}**"
    )


# ============================================================
# QUESTION ANSWERING
# ============================================================

if st.session_state.get(
    "pdf_processed",
    False
):

    st.divider()

    st.header("💬 Ask a Question")

    question = st.text_input(
        "Enter your question",
        placeholder="Example: What is the main topic of this document?"
    )

    if st.button(
        "🔍 Ask Question",
        type="primary"
    ):

        if not question.strip():

            st.warning(
                "Please enter a question."
            )

        else:

            # ------------------------------------------------
            # SEARCH DOCUMENT
            # ------------------------------------------------

            with st.spinner(
                "Searching the document..."
            ):

                try:

                    relevant_chunks = search_documents(
                        question,
                        number_of_results=5
                    )

                except Exception as error:

                    st.error(
                        "Error while searching the document."
                    )

                    st.exception(error)

                    st.stop()

            if not relevant_chunks:

                st.warning(
                    "No relevant information was found."
                )

            else:

                context = "\n\n".join(
                    relevant_chunks
                )

                # ------------------------------------------------
                # GENERATE ANSWER
                # ------------------------------------------------

                with st.spinner(
                    "Generating answer..."
                ):

                    try:

                        answer = generate_answer(
                            question,
                            context
                        )

                    except Exception as error:

                        st.error(
                            "Could not connect to Ollama."
                        )

                        st.warning(
                            "Make sure Ollama is installed "
                            "and the llama3.2 model is available."
                        )

                        st.code(
                            "ollama pull llama3.2"
                        )

                        st.exception(error)

                        st.stop()

                # ------------------------------------------------
                # DISPLAY ANSWER
                # ------------------------------------------------

                st.subheader("🤖 Answer")

                st.write(answer)

                # ------------------------------------------------
                # DISPLAY SOURCES
                # ------------------------------------------------

                with st.expander(
                    "📖 View retrieved document sections"
                ):

                    for index, chunk in enumerate(
                        relevant_chunks,
                        start=1
                    ):

                        st.markdown(
                            f"**Source {index}**"
                        )

                        st.write(chunk)

                        st.divider()


# ============================================================
# INITIAL MESSAGE
# ============================================================

else:

    st.info(
        "👆 Upload a PDF and click **Process PDF** "
        "to start asking questions."
    )
