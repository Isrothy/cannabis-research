import gplay from "google-play-scraper";
import connectDB from "./db.js";
import { GoogleApp } from "./models/App.js";
import mongoose from "mongoose";

const dbUri = process.env.MONGO_URI;
await connectDB(dbUri);

async function main() {
  const apps = await GoogleApp.find({
    reviewList: {
      $exists: false,
    },
    $or: [
      { predictedLabel: { $eq: 1 } },
      { "keywordCounts.cannabis": { $gt: 0 } },
      { "keywordCounts.marijuana": { $gt: 0 } },
    ],
  });
  console.log(`Found ${apps.length} apps in google play collection.`);

  for (let i = 0; i < apps.length; i++) {
    const app = apps[i];
    const num_reviews = app.reviews;
    console.log(`Fetching reviews for "${app.title}":`);
    try {
      let list = [];
      let token = null;
      while (true) {
        const ret = await gplay.reviews({
          appId: app.appId,
          lang: "en",
          sort: gplay.sort.HELPFULNESS,
          num: num_reviews,
          paginate: true,
          nextPaginationToken: token,
        });
        token = ret.nextPaginationToken;
        const data = ret.data;
        if (!token || data.length == 0) {
          break;
        }
        list = list.concat(data);
        console.log(`Fetched ${data.length} reviews for "${app.title}":`);
      }
      console.log(`Fetched all reviews for "${app.title}":`);

      const result = await GoogleApp.updateOne(
        { _id: app._id },
        { $set: { reviewList: list } },
      );
      console.log(`Prepared update for "${app.title}".`);
      console.log(
        `Matched ${result.matchedCount} document(s) and modified ${result.modifiedCount} document(s).`,
      );
    } catch (err) {
      console.error(`Error updating app "${app.title}": ${err}`);
    }
  }
}

await main();
await mongoose.disconnect();
