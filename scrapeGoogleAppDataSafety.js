import gplay from "google-play-scraper";
import connectDB from "./db.js";
import { GoogleApp } from "./models/App.js";
import mongoose from "mongoose";

const dbUri = process.env.MONGO_URI;
await connectDB(dbUri);

async function main() {
  const apps = await GoogleApp.find({ dataSafety: { $exists: false } });
  console.log(`Found ${apps.length} apps in google play collection.`);

  const bulkBatchSize = 20;
  let bulkOps = [];

  for (let i = 0; i < apps.length; i++) {
    const app = apps[i];
    try {
      const data = await gplay.datasafety({
        appId: app.appId,
        lang: "en",
      });
      console.log(`Fetched ratings for "${app.title}":`);

      bulkOps.push({
        updateOne: {
          filter: { _id: app._id },
          update: {
            $set: {
              dataSafety: data,
            },
          },
        },
      });
      console.log(`Prepared update for "${app.title}".`);

      if (bulkOps.length === bulkBatchSize) {
        const bulkResult = await GoogleApp.bulkWrite(bulkOps);
        console.log(`Bulk update result for batch: `, bulkResult);
        bulkOps = [];
      }
    } catch (err) {
      console.error(`Error updating app "${app.title}": ${err}`);
    }
  }
  if (bulkOps.length != 0) {
    const bulkResult = await GoogleApp.bulkWrite(bulkOps);
    console.log(`Bulk update result for batch: `, bulkResult);
  }
}

await main();
await mongoose.disconnect();
