import { ChromaClient } from "chromadb";
import "./utils/env.js";
import { makeRequire } from "./utils/utils.js";
import { InferenceClient } from "@huggingface/inference";
import africanFoods from "./data/african_food_dataset.json" assert { type: "json" };

const require = makeRequire(import.meta.url);
const foods = require("./data/FoodDataSet.cjs");

const { HF_TOKEN } = process.env;

const collectionName = "foods";
let foodItems = [];

const client = new ChromaClient();
const hf = new InferenceClient(HF_TOKEN);

// Merge all food items
function mergeFoods() {
  if (!foodItems.length) {
    foodItems = [...foods, ...africanFoods];
  }
  return foodItems;
}

// generate embeddings
async function generateEmbeddings(texts) {
  const results = await hf.featureExtraction({
    model: "sentence-transformers/all-MiniLM-L12-v2",
    inputs: texts,
  });
  return results;
}

// classify text and add labels
async function classifyText(text, labels) {
  const response = await hf.zeroShotClassification({
    model: "facebook/bart-large-mnli",
    inputs: text,
    parameters: {
      candidate_labels: labels,
    },
  });

  console.log("Classification response: ", response);

  return response;
}

// extract criteria
async function extractFilterCriteria(query) {
  const criteria = { diet: null, cuisine: null };

  const dietLabels = [
    "vegan",
    "non-vegan",
    "vegetarian",
    "non-vegetarian",
    "pescatarian",
    "omnivore",
    "paleo",
    "ketogenic",
  ];
  const cuisineLabels = ["yoruba", "nigerian", "chinese", "indian", "japanese"];

  const dietResult = await classifyText(query, dietLabels);
  const highestDietScoreLabel = dietResult[0].label;
  const dietScore = dietResult[0].score;

  // Only apply diet criteria if the score is very high (e.g., > 0.8)
  if (dietScore > 0.8) {
    criteria.diet = highestDietScoreLabel;
  } else {
    const cuisineResult = await classifyText(query, cuisineLabels);
    const highestCuisineScoreLabel = cuisineResult[0].label;
    const cuisineScore = cuisineResult[0].score;

    // Only apply cuisine criteria if the score is very high (e.g., > 0.8)
    if (cuisineScore > 0.8) {
      criteria.cuisine = highestCuisineScoreLabel;
    }
  }
  console.log("Extracted Filter Criteria:", criteria);
  return criteria;
}

// Search for similar foods
async function performSimilaritySearch(collection, queryTerm, filterCriteria) {
  try {
    const queryEmbedding = await generateEmbeddings([queryTerm]);
    console.log(filterCriteria);
    const results = await collection.query({
      collection: collectionName,
      queryEmbeddings: queryEmbedding,
      n: 5,
    });

    if (!results || results.length === 0) {
      console.log(`No food items found similar to "${queryTerm}"`);
      return [];
    }

    let topFoodItems = results.ids[0]
      .map((id, index) => {
        return {
          id,
          score: results.distances[0][index],
          food_name: foodItems.find((item) => item.food_id.toString() === id)
            .food_name,
          food_description: foodItems.find(
            (item) => item.food_id.toString() === id
          ).food_description,
        };
      })
      .filter(Boolean);
    return topFoodItems.sort((a, b) => a.score - b.score);
  } catch (error) {
    console.error("Error during similarity search:", error);
    return [];
  }
}

async function main() {
  try {
    const collection = await client.getOrCreateCollection({
      name: collectionName,
    });

    foodItems = mergeFoods();

    // Create unique IDs for each item
    const uniqueIds = new Set();
    foodItems.forEach((food, index) => {
      while (uniqueIds.has(food.food_id.toString())) {
        food.food_id = `${food.food_id}_${index}`;
      }
      uniqueIds.add(food.food_id.toString());
    });

    const foodTexts = foodItems.map(
      (food) =>
        `${food.food_name}. ${
          food.food_description
        }. Ingredients: ${food.food_ingredients.join(", ")}`
    );
    const embeddingsData = await generateEmbeddings(foodTexts);
    await collection.add({
      ids: foodItems.map((food) => food.food_id.toString()),
      documents: foodTexts,
      embeddings: embeddingsData,
    });

    const query = "I want to eat fufu with a nice soup for lunch";

    const filterCriteria = await extractFilterCriteria(query);

    const initialResults = await performSimilaritySearch(
      collection,
      query,
      filterCriteria
    );

    initialResults.slice(0, 5).forEach((item, index) => {
      console.log(
        `Top ${index + 1} Recommended Food Name ==>, ${
          item.food_name
        }, ranking ${item.score}`
      );
    });
  } catch (error) {
    console.error("An error occured: ", error);
  }
}

main();
