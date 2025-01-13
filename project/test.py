from main import Engine, Model
import pytest

@pytest.mark.filterwarnings("ignore")
@pytest.mark.parametrize("query", ["Mona Lisa", "Number Theory", "Species", "Astrophysics", "Ecosystem"])
def test_engine_search(capfd, query):

	e = Engine()
	e.search(query)

	out, err = capfd.readouterr()

	assert out.lower().count(query.lower()) > 1