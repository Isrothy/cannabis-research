import connectDB from "./db.js";
import store from "app-store-scraper";
import { AppleApp } from "./models/App.js";
import mongoose from "mongoose";

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

async function updateAppleAppRatings() {
  try {
    const apps = await AppleApp.find({ ratings: { $exists: false } });
    console.log(`Found ${apps.length} apps in AppleApps collection.`);

    const bulkBatchSize = 20;
    let bulkOps = [];

    for (let i = 0; i < apps.length; i++) {
      const app = apps[i];
      try {
        const data = await store.app({
          appId: app.appId,
          ratings: true,
        });
        const ratingData = await store.ratings({
          id: data.id,
        });
        console.log(`Fetched ratings for "${app.title}":`, ratingData.ratings);

        bulkOps.push({
          updateOne: {
            filter: { _id: app._id },
            update: {
              $set: {
                ratings: ratingData.ratings,
                histogram: ratingData.histogram,
              },
            },
          },
        });
        console.log(`Prepared update for "${app.title}".`);

        // If we've reached the batch size, execute the bulkWrite operation
        if (bulkOps.length === bulkBatchSize) {
          const bulkResult = await AppleApp.bulkWrite(bulkOps);
          console.log(
            `Bulk update result for batch ending at index ${i}:`,
            bulkResult,
          );
          bulkOps = []; // reset bulkOps for next batch
        }
      } catch (err) {
        console.error(`Error updating app "${app.title}":`);
      }

      await new Promise((resolve) => setTimeout(resolve, 1000));
    }

    // Process any remaining operations in the last batch
    if (bulkOps.length > 0) {
      const bulkResult = await AppleApp.bulkWrite(bulkOps);
      console.log("Bulk update result for final batch:", bulkResult);
    }

    console.log("Finished updating ratings for all apps.");
  } catch (err) {
    console.error("Error fetching Apple apps:", err);
  } finally {
    await mongoose.connection.close();
    process.exit(0);
  }
}

const dbUri = process.env.MONGO_URI;
if (!dbUri) {
  console.error("MONGO_URI environment variable is not set.");
  process.exit(1);
}

await connectDB(dbUri);
updateAppleAppRatings();
