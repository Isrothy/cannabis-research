import connectDB from "./db.js";
import gplay from "google-play-scraper";
import appleScraper from "app-store-scraper";
import { GoogleApp, AppleApp } from "./models/App.js";

const args = {};
for (let i = 2; i < process.argv.length; i++) {
  const arg = process.argv[i];
  if (arg.startsWith("--")) {
    const key = arg.slice(2);
    const nextArg = process.argv[i + 1];
    if (nextArg && !nextArg.startsWith("--")) {
      args[key] = nextArg;
      i++;
    } else {
      args[key] = true;
    }
  }
}

const searchTerm = args.term || "cannabis";
const numApps = parseInt(args.num) || 20;
const source = (args.source || "google").toLowerCase();

const dbUri = process.env.MONGO_URI;

let scraper;
let AppModel;

if (source === "google") {
  scraper = gplay;
  AppModel = GoogleApp;
} else if (source === "apple") {
  scraper = appleScraper;
  AppModel = AppleApp;
} else {
  console.error("Unknown source specified. Use 'google' or 'apple'.");
  process.exit(1);
}

await connectDB(dbUri);

async function main() {
  try {
    const searchResults = await scraper.search({
      term: searchTerm,
      num: numApps,
    });
    console.log(`Found ${searchResults.length} apps.`);

    for (let i = 0; i < searchResults.length; i++) {
      const rank = i + 1;
      const result = searchResults[i];
      console.log(`Processing app (${rank}): ${result.title}`);
      try {
        let appDetail;
        if (source === "google") {
          appDetail = await scraper.app({ appId: result.appId });
        } else if (source === "apple") {
          appDetail = await scraper.app({ id: result.id });
        }

        const newSearchTerm = { term: searchTerm, rank: rank };

        let existing = await AppModel.findOne({ appId: appDetail.appId });
        if (existing) {
          const existingIndex = existing.searchTerms.findIndex(
            (st) => st.term === searchTerm,
          );
          if (existingIndex === -1) {
            existing.searchTerms.push(newSearchTerm);
            await existing.save();
            console.log(
              `Updated app ${existing.title} with new search term and rank.`,
            );
          } else {
            if (existing.searchTerms[existingIndex].rank !== rank) {
              existing.searchTerms[existingIndex].rank = rank;
              await existing.save();
              console.log(
                `Updated rank for app ${existing.title} on search term.`,
              );
            } else {
              console.log(
                `App ${existing.title} already contains the search term with same rank.`,
              );
            }
          }
        } else {
          appDetail.searchTerms = [newSearchTerm];
          const newApp = new AppModel(appDetail);
          await newApp.save();
          console.log(`Inserted new app: ${newApp.title}`);
        }
      } catch (innerErr) {
        console.error("Error processing app:", innerErr);
      }
    }
    process.exit(0);
  } catch (err) {
    console.error("Error during scraping:", err);
    process.exit(1);
  }
}

main();
