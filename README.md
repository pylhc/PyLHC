# Repository Coverage

[Full report](https://htmlpreview.github.io/?https://github.com/pylhc/PyLHC/blob/python-coverage-comment-action-data/htmlcov/index.html)

| Name                                       |    Stmts |     Miss |   Cover |   Missing |
|------------------------------------------- | -------: | -------: | ------: | --------: |
| pylhc/\_\_init\_\_.py                      |        8 |        0 |    100% |           |
| pylhc/bpm\_calibration.py                  |       26 |        1 |     96% |       130 |
| pylhc/bsrt\_analysis.py                    |      191 |       26 |     86% |106-118, 133, 137, 141, 145, 218, 256, 258, 342, 344, 406, 408, 452, 454, 461 |
| pylhc/bsrt\_logger.py                      |       59 |       59 |      0% |    13-103 |
| pylhc/calibration/\_\_init\_\_.py          |        0 |        0 |    100% |           |
| pylhc/calibration/beta.py                  |       79 |        0 |    100% |           |
| pylhc/calibration/dispersion.py            |       61 |        0 |    100% |           |
| pylhc/constants/\_\_init\_\_.py            |        0 |        0 |    100% |           |
| pylhc/constants/calibration.py             |       10 |        0 |    100% |           |
| pylhc/constants/forced\_da\_analysis.py    |       77 |       19 |     75% |33, 37, 41, 52, 71-72, 76, 93, 97, 101, 105, 124, 128, 132, 136, 145, 149, 153, 157 |
| pylhc/constants/general.py                 |       13 |        2 |     85% |    35, 40 |
| pylhc/constants/kickgroups.py              |       26 |       26 |      0% |      8-57 |
| pylhc/constants/machine\_settings\_info.py |       19 |       19 |      0% |      9-35 |
| pylhc/data\_extract/\_\_init\_\_.py        |        0 |        0 |    100% |           |
| pylhc/data\_extract/lsa.py                 |      176 |      136 |     23% |45-51, 64-74, 90-95, 111-124, 148-170, 195-228, 240-245, 266-279, 293-321, 325-334, 346-366, 382-390, 405-416 |
| pylhc/data\_extract/timber.py              |       15 |       15 |      0% |      8-37 |
| pylhc/forced\_da\_analysis.py              |      625 |      531 |     15% |196-208, 216-222, 369-427, 435-441, 453-464, 469-470, 478-488, 492, 496, 500, 504-507, 517-549, 554-561, 566-588, 601-636, 641-643, 647-657, 661-702, 706-739, 746-756, 761-779, 784-791, 795-805, 812-817, 821-839, 843-859, 866-889, 897, 902, 907, 912, 916-937, 942-950, 955-960, 966-974, 979-981, 987-998, 1003-1008, 1013-1045, 1056-1127, 1131-1185, 1193-1353, 1357-1365, 1378-1392, 1397-1399, 1404-1420, 1430-1437, 1441-1447, 1454 |
| pylhc/kickgroups.py                        |      190 |      190 |      0% |    60-552 |
| pylhc/lsa\_to\_madx.py                     |      126 |       51 |     60% |137-139, 157-158, 162-163, 205-206, 229, 248, 287-288, 299-330, 334-383, 390 |
| pylhc/machine\_settings\_info.py           |      172 |      172 |      0% |    51-578 |
|                                  **TOTAL** | **1873** | **1247** | **33%** |           |


## Setup coverage badge

Below are examples of the badges you can use in your main branch `README` file.

### Direct image

[![Coverage badge](https://raw.githubusercontent.com/pylhc/PyLHC/python-coverage-comment-action-data/badge.svg)](https://htmlpreview.github.io/?https://github.com/pylhc/PyLHC/blob/python-coverage-comment-action-data/htmlcov/index.html)

This is the one to use if your repository is private or if you don't want to customize anything.

### [Shields.io](https://shields.io) Json Endpoint

[![Coverage badge](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/pylhc/PyLHC/python-coverage-comment-action-data/endpoint.json)](https://htmlpreview.github.io/?https://github.com/pylhc/PyLHC/blob/python-coverage-comment-action-data/htmlcov/index.html)

Using this one will allow you to [customize](https://shields.io/endpoint) the look of your badge.
It won't work with private repositories. It won't be refreshed more than once per five minutes.

### [Shields.io](https://shields.io) Dynamic Badge

[![Coverage badge](https://img.shields.io/badge/dynamic/json?color=brightgreen&label=coverage&query=%24.message&url=https%3A%2F%2Fraw.githubusercontent.com%2Fpylhc%2FPyLHC%2Fpython-coverage-comment-action-data%2Fendpoint.json)](https://htmlpreview.github.io/?https://github.com/pylhc/PyLHC/blob/python-coverage-comment-action-data/htmlcov/index.html)

This one will always be the same color. It won't work for private repos. I'm not even sure why we included it.

## What is that?

This branch is part of the
[python-coverage-comment-action](https://github.com/marketplace/actions/python-coverage-comment)
GitHub Action. All the files in this branch are automatically generated and may be
overwritten at any moment.