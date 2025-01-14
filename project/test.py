from main import Engine, Model
import pytest

query_definitions = [
    "Microeconomics", "individuals", "scarce resources", "interactions", "Macroeconomics",
    "price index", "normalized average", "price relatives",  "performance",
    "goods", "organism", "functions", "individual", "definition", 
    "problems", "solve", "Animals", "Plants", "eukaryotes", "photosynthetic",
    "energy", "sunlight", "Atoms", "chemical elements", "human past",
    "atom", "geometry", "mathematics", "behavior", "study", "goods or services",
    "region", "definition", "problems", "Mona Lisa", "number theory", "species", 
    "astrophysics", "ecosystem", "mathematical logic", "decorative arts",
    "Money", "payment", "goods and services", "Archaeology", "History", 
    "Art", "Ballet", "performance dance", "France", "Russia",
    "Renaissance", "Middle Ages", "modernity", "historical dance",
    "Renaissance music", "Marquetry", "Intarsia", "Inlay", "decorative arts",
    "Mona Lisa", "half-length portrait", "archetypal masterpiece",
    "Italian", "decorative arts", "beautiful and functional",
    "Relativity", "space and time", "special relativity", "general relativity",
    "acceleration", "Optics", "behaviour", "properties of light", "Acoustics",
    "Particle physics", "interactions", "Nuclear physics", "nuclear reactions", "Astrophysics",
    "astronomical objects", "phenomena", "Biology", "scientific study", "life",
    "Anatomy", "morphology", "Ecology", "relationships", "environment", "Genetics", "genes",
    "Zoology", "scientific study of animals", "Botany", "Chemistry",
    "behavior of matter", "chemical elements", "Physical chemistry", "Organic chemistry",
]

@pytest.mark.filterwarnings("ignore")
@pytest.mark.parametrize("query", query_definitions)
def test_engine_search_definitions(capfd, query):

	e = Engine()
	e.search(query)

	out, err = capfd.readouterr()

	assert out.lower().count(query.lower()) > 1

query_questions = [
    "What is the concept of Microeconomics?",
    "What studies nuclear reaction in atoms?",
]

query_answers = [
    "Microeconomics focuses on the study of individual markets, sectors, or industries as opposed to the economy as a whole, which is studied in macroeconomics.",
    "Nuclear physics",
]

@pytest.mark.filterwarnings("ignore")
@pytest.mark.parametrize("question, answer", zip(query_questions, query_answers))
def test_engine_search_questions(capfd, question, answer):

	e = Engine()
	e.search(question)

	out, err = capfd.readouterr()

	assert out.lower().count(answer.lower()) > 0