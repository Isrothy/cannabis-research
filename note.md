# Note

## Search

Using [google-play-scraper](https://github.com/facundoolano/google-play-scraper) and [app-store-scraper](https://github.com/facundoolano/app-store-scraper) to download app metadata

Using keywords:

- cannabis
- marijuana
- weed
- grass
- herb
- chronic
- ganja
- reefer
- hemp
- bud
- pot
- dope

Only download the first 250 apps.

## Filtering

Use **facebook/bart-large-mnli** to identify whether the apps are cannabis related.
Focus on apps with a score higher than 0.7

## Result

|                                                                                  | Apple App Store | Google Play | Note                                                                                |
| -------------------------------------------------------------------------------- | --------------- | ----------- | ----------------------------------------------------------------------------------- |
| Total Apps                                                                       | 1846            | 2052        |                                                                                     |
| Total Filtered Apps                                                              | 358             | 244         |                                                                                     |
| With search term "cannabis" or "marijuana"                                       | 267             | 159         |                                                                                     |
| With search term "weed" and do not have search term "cannabis" or "marijuana"    | 28              | 8           |                                                                                     |
| With search term "grass" and do not have search term "cannabis" or "marijuana"   | 3               | 7           |                                                                                     |
| With search term "herb" and do not have search term "cannabis" or "marijuana"    | 5               | 3           |                                                                                     |
| With search term "chronic" and do not have search term "cannabis" or "marijuana" | 4               | 7           |                                                                                     |
| With search term "ganja" and do not have search term "cannabis" or "marijuana"   | 17              | 3           |                                                                                     |
| With search term "reefer" and do not have search term "cannabis" or "marijuana"  | 8               | 9           |                                                                                     |
| With search term "hemp" and do not have search term "cannabis" or "marijuana"    | 38              | 6           |                                                                                     |
| With search term "bud" and do not have search term "cannabis" or "marijuana"     | 15              | 24          | Many of them on google play are not cannabis related but misclassified by the model |
| With search term "pot" and do not have search term "cannabis" or "marijuana"     | 3               | 30          | Many of them on google play are not cannabis related but misclassified by the model |
| With search term "dope" and do not have search term "cannabis" or "marijuana"    | 2               | 7           |                                                                                     |
| With search term "cannabis", "marijuana", "weed" and "hemp"                      | 324             | 172         |                                                                                     |

## TODO

1.   Shared google sheet
2.   Find a threshold for best F1 score
