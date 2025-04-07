import connectDB from "./db.js";
import store from "app-store-scraper";
import { AppleApp } from "./models/App.js";
import mongoose from "mongoose";

async function main() {
  try {
    const apps = await AppleApp.find({ id: { $exists: false } });
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
        console.log(`Fetched id for "${app.title}"`);

        bulkOps.push({
          updateOne: {
            filter: { _id: app._id },
            update: {
              $set: {
                id: data.id,
              },
            },
          },
        });
        console.log(`Prepared update for "${app.title}".`);

        if (bulkOps.length === bulkBatchSize) {
          const bulkResult = await AppleApp.bulkWrite(bulkOps);
          console.log(
            `Bulk update result for batch ending at index ${i}:`,
            bulkResult,
          );
          bulkOps = [];
        }
      } catch (err) {
        console.error(`Error updating app "${app.title}":`);
      }

      await new Promise((resolve) => setTimeout(resolve, 1000));
    }

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
main();
