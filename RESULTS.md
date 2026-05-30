# RESULTS — Task 4

## Live URL
Local only: http://127.0.0.1:8001

## Setup
- Embedding model used: `insightface` buffalo_l
- Face detector / aligner used: InsightFace face detector + aligned embedding pipeline
- Qdrant: local instance on `http://localhost:6333`

## Enrolled identities
| id | source |
|----|--------|
| id0 | id0_Atal_Bihari_Vajpayee.jpg |
| id1 | id1_David_Trimble.jpg |
| id2 | id2_Donald_Rumsfeld.jpg |
| id3 | id3_Cherie_Blair.jpg |
| id4 | id4_John_Negroponte.jpg |

## Query result
- Query image: query_should_match_id0.jpg
- Top match returned: `id0`
- Cosine score: 0.62122166
- Correct? (yes/no): yes

## Hard negative (look-alike)
- The two similar identities: `id0` and `id4`
- Cosine score between them: 0.086942
- Your match/no-match threshold: 0.4
- Why this threshold (which error is worse here and why): With the genuine match at 0.62 and the strongest non-match under 0.16, a threshold around 0.4 avoids false accepts while still accepting real matches. In face matching, false accepts are generally worse than false rejects for security.

## Search latency
- Approx ms per /search call: 2048.09

## Anything that broke / would improve with more time
- Would add explicit match/no-match thresholding, batch enrollment, and multi-face query support.

