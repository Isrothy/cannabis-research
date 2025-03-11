import { Schema, model } from "mongoose";

const AppSchema = new Schema(
  {
    appId: { type: String, required: true, unique: true },
    title: { type: String },
    searchTerms: {
      type: [
        {
          term: { type: String },
          rank: { type: Number },
        },
      ],
      default: [],
    },
    // Any additional fields from the scraped data will be stored as well.
  },
  { timestamps: true, strict: false }, // strict:false allows extra fields to be saved
);

export const GoogleApp = model("GoogleApp", AppSchema, "googleApps");
export const AppleApp = model("AppleApp", AppSchema, "appleApps");
