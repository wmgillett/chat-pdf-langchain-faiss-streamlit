# tests/test_app.py
import pytest
import pickle
from unittest.mock import Mock
from app import process_text, get_embeddings, retrieve_docs, generate_response

# Mock the PdfReader object for testing
class MockPdfReader:
    class MockPage:
        def extract_text(self):
            return "This is a test text"

    @property
    def pages(self):
        return [self.MockPage() for _ in range(3)]

# Define a fixture for mock PDF
@pytest.fixture
def mock_pdf():
    pdf = Mock()
    pdf.name = "test.pdf"
    return pdf

# Define a fixture for chunks
@pytest.fixture
def chunks():
    return ["This is a test text"]

def test_process_text(mocker, mock_pdf):
    mocker.patch('app.PdfReader', return_value=MockPdfReader())
    chuck_size = 24
    chuck_overlap = 0
    chunks = process_text(mock_pdf, chuck_size, chuck_overlap)
    assert len(chunks) == 3
    assert chunks[0] == "This is a test textThis"

def test_get_embeddings(mocker, mock_pdf, chunks):
    mocker.patch('os.path.exists', return_value=False)
    mocker.patch('app.OpenAIEmbeddings')
    mocker.patch('app.FAISS.from_texts')
    mocker.patch('pickle.dump')
    vector_store = get_embeddings(chunks, mock_pdf)
    assert vector_store is not None

def test_retrieve_docs(mocker):
    vector_store = Mock()
    vector_store.similarity_search.return_value = ["Doc 1", "Doc 2", "Doc 3"]
    question = "What is the test question?"
    docs = retrieve_docs(question, vector_store)
    assert len(docs) == 3

def test_generate_response(mocker):
    mocker.patch('app.OpenAI')
    mocker.patch('app.load_qa_chain')
    mocker.patch('app.get_openai_callback')
    docs = ["Doc 1", "Doc 2", "Doc 3"]
    question = "What is the test question?"
    response = generate_response(docs, question)
    assert response is not None
