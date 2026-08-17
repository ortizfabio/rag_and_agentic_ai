from ibm_watsonx_ai.foundation_models import ModelInference
from ibm_watsonx_ai.foundation_models.extensions.langchain import WatsonxLLM
from ibm_watsonx_ai.metanames import EmbedTextParamsMetaNames
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams

try:
    from langchain_ibm import WatsonxEmbeddings
except ImportError:
    WatsonxEmbeddings = None

def llm():
    model_id = 'mistralai/mistral-small-3-1-24b-instruct-2503'

    parameters = {
        GenParams.MAX_NEW_TOKENS: 256,  # this controls the maximum number of tokens in the generated output
        GenParams.TEMPERATURE: 0.5, # this randomness or creativity of the model's responses
    }

    credentials = {
        "url": "https://us-south.ml.cloud.ibm.com"
    }


    project_id = "skills-network"

    model = ModelInference(
        model_id=model_id,
        params=parameters,
        credentials=credentials,
        project_id=project_id
    )

    mixtral_llm = WatsonxLLM(model = model)
    return mixtral_llm



def watsonx_embedding():
    if WatsonxEmbeddings is None:
        raise ImportError("WatsonxEmbeddings not available. Install langchain-ibm: pip install langchain-ibm")
    
    embed_params = {
        EmbedTextParamsMetaNames.TRUNCATE_INPUT_TOKENS: 3,
        EmbedTextParamsMetaNames.RETURN_OPTIONS: {"input_text": True},
    }

    watsonx_embedding = WatsonxEmbeddings(
        #        model_id="ibm/slate-125m-english-rtrvr-v2",
        model_id="ibm/granite-embedding-278m-multilingual",
        url="https://us-south.ml.cloud.ibm.com",
        project_id="skills-network",
        params=embed_params,
    )
    return watsonx_embedding

