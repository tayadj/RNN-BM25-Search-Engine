from main import Engine, Model
import pytest

@pytest.mark.filterwarnings("ignore")
def test_engine_search_vocabulary(capfd):

    engine = Engine()
    simulation = lambda capfd, query, engine: (engine.search(query), out := capfd.readouterr())[1][0].lower().count(query.lower()) > 1
    vocabulary = engine.model.vocabulary

    total = engine.model.vocabulary_size
    passed = sum(1 for query in vocabulary if simulation(capfd, query, engine))

    assert (passed / total) >= 0.95