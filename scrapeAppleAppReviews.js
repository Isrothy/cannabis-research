import store from "app-store-scraper";
import connectDB from "./db.js";
import { AppleApp } from "./models/App.js";
import mongoose from "mongoose";

// store
//   .reviews({
//     appId: "com.metamoki.weed",
//     sort: store.sort.HELPFUL,
//     page: 10,
//   })
//   .then(console.log)
//   .catch(console.log);

const dbUri = process.env.MONGO_URI;
await connectDB(dbUri);

async function main() {
  const apps = await AppleApp.find({
    helpfulReviewList: { $exists: false },
    recentReviewList: { $exists: false },
    $or: [
      { predictedLabel: { $eq: 1 } },
      { "keywordCounts.cannabis": { $gt: 0 } },
      { "keywordCounts.marijuana": { $gt: 0 } },
    ],
  });
  console.log(`Found ${apps.length} apps in the Apple App collection.`);

  for (let i = 0; i < apps.length; i++) {
    const app = apps[i];
    console.log(`Fetching reviews for "${app.title}" (AppId: ${app.appId})`);
    try {
      let helpfulReviews = [];
      let recentReviews = [];

      let page = 1;
      while (page <= 10) {
        const reviews = await store.reviews({
          appId: app.appId,
          sort: store.sort.HELPFUL,
          page: page,
        });
        if (!reviews || reviews.length === 0) break;
        helpfulReviews = helpfulReviews.concat(reviews);
        console.log(
          `Fetched ${reviews.length} HELPFUL reviews on page ${page} for "${app.title}"`,
        );
        page++;
        await new Promise((resolve) => setTimeout(resolve, 1000));
      }

      page = 1;
      while (page <= 10) {
        const reviews = await store.reviews({
          appId: app.appId,
          sort: store.sort.RECENT,
          page: page,
        });
        if (!reviews || reviews.length === 0) break;
        recentReviews = recentReviews.concat(reviews);
        console.log(
          `Fetched ${reviews.length} RECENT reviews on page ${page} for "${app.title}"`,
        );
        page++;
        await new Promise((resolve) => setTimeout(resolve, 1000));
      }

      const result = await AppleApp.updateOne(
        { _id: app._id },
        {
          $set: {
            helpfulReviewList: helpfulReviews,
            recentReviewList: recentReviews,
          },
        },
      );
      console.log(
        `Updated "${app.title}": Matched ${result.matchedCount} and Modified ${result.modifiedCount}`,
      );
    } catch (err) {
      console.error(`Error updating app "${app.title}": ${err}`);
    }
  }
}

await main();
await mongoose.disconnect();
