const { InferenceClient } = require("@huggingface/inference");

const hf = new InferenceClient("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");

const text = "Let's use a hugging face AI model";

const getEmbeddings = async () => {
  try {
    let embeddings = await convertTextToEmbedding(text);
    console.log(embeddings);
  } catch (err) {
    console.error("Error getting embeddings:", err);
  }
};

const convertTextToEmbedding = async (text) => {
  try {
    const result = await hf.featureExtraction({
      model: "sentence-transformers/all-MiniLM-L6-v2",
      inputs: text,
    });
    return result;
  } catch (err) {
    console.error("Error converting text to embeddings:", err);
    throw err;
  }
};

// Call the async function to get the embeddings
getEmbeddings();
