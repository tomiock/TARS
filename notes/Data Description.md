

## Project and Task Management Data Structure

This document outlines the data structure used for managing projects, tasks, translators, and clients within a translation workflow.

### Project Table

* **PROJECT_ID:** Project code (extra info, not likely essential).
* **PM:** Responsible project management team.
* **TASK_ID:** Task code.
* **START:** Task start date.
* **END:** Theoretical task delivery date (can be used to compare with DELIVERED date to track delays).
* **TASK_TYPE:** Type of task. Specific considerations for each type:
    * **DTP:** Desktop-Publishing tasks.
    * **Engineering:** Engineering tasks, such as file conversions, encoding, etc.
    * **LanguageLead:** Linguistic management tasks. Assigned to highly experienced, high-quality individuals who regularly work on the project.
    * **Management:** General management tasks.
    * **Miscellaneous:** Various linguistically-oriented tasks.
    * **PostEditing:** Post-editing tasks. Similar to Translation, but the required TRANSLATOR skills are slightly different.
    * **ProofReading:** Complete review of a Translation or PostEditing. This task always follows a Translation or PostEditing. The assigned TRANSLATOR must have more experience than the person who performed the initial step.
    * **Spotcheck:** Partial review of a Translation or PostEditing. This task always follows a Translation or PostEditing. The assigned TRANSLATOR must have more experience than the person who performed the initial step.
    * **TEST:** A test required to gain access to work with a client. It should be prioritized to the TRANSLATOR with the most experience and quality for the client or subject matter, regardless of price, but considering the deadline.
    * **Training:** TRANSLATOR experience and quality are not critical.
    * **Translation:** Translation task. Translator quality can be slightly lower than required if the ProofReading (not Spotcheck) is performed by a higher-quality translator. If a Spotcheck is performed, the quality must meet the requirement.
* **SOURCE_LANG:** Source language.
* **TARGET_LANG:** Target language.
* **TRANSLATOR:** Translator assigned to the task.
* **ASSIGNED:** Time of assignment (advance notice) to the TRANSLATOR (see Kanban system: https://en.wikipedia.org/wiki/Kanban).
* **READY:** Time when the TRANSLATOR is notified to begin.
* **WORKING:** Time when the TRANSLATOR starts the task.
* **DELIVERED:** Time when the TRANSLATOR delivers the task.
* **RECEIVED:** Time when the PM receives the task.
* **CLOSE:** Time when the PM marks the task as completed.
* **FORECAST:** Estimated hours of work.
* **HOURLY_RATE:** Hourly rate for the specific task.
* **COST:** Total cost of the task.
* **QUALITY_EVALUATION:** Quality control evaluation.
* **MANUFACTURER:** Client.
* **MANUFACTURER_SECTOR:** Client categorization level 1.
* **MANUFACTURER_INDUSTRY_GROUP:** Client categorization level 2.
* **MANUFACTURER_INDUSTRY:** Client categorization level 3.
* **MANUFACTURER_SUBINDUSTRY:** Client categorization level 4.

### Schedules Table

* **NAME:** TRANSLATOR's name.
* **START:** Start time of their workday.
* **END:** End time of their workday.
* **MON:** Works on Monday? (1 yes, 0 no).
* **TUES:** Works on Tuesday? (1 yes, 0 no).
* **WED:** Works on Wednesday? (1 yes, 0 no).
* **THURS:** Works on Thursday? (1 yes, 0 no).
* **FRI:** Works on Friday? (1 yes, 0 no).
* **SAT:** Works on Saturday? (1 yes, 0 no).
* **SUN:** Works on Sunday? (1 yes, 0 no).

### Clients Table

* **CLIENT_NAME:** Client's name.
* **SELLING_HOURLY_PRICE:** Hourly price charged to the client.
* **MIN_QUALITY:** Minimum quality expected from TRANSLATORs.
* **WILDCARD:** Which condition can be overridden if all conditions cannot be met.

### TranslatorsCost+Pairs Table

* **TRANSLATOR:** Translator's name.
* **SOURCE_LANG:** Source language.
* **TARGET_LANG:** Target language.
* **HOURLY_RATE:** Cost per hour.

### Other Considerations

* Translator experience should be assessed based on the hours they have translated for a specific client, a client type, or a task type.

## Table with codifications

| Data Label                               | Type of Data | Info |
| ---------------------------------------- | ------------ | ---- |
| PROJECT_ID                               |              |      |
| PM                                       |              |      |
| TASK_ID                                  |              |      |
| START                                    |              |      |
| END                                      |              |      |
| TASK_TYPE                                |              |      |
| SOURCE_LANG                              |              |      |
| TARGET_LANG                              |              |      |
| TRANSLATOR                               |              |      |
| ASSIGNED                                 |              |      |
| READY                                    |              |      |
| WORKING                                  |              |      |
| DELIVERED                                |              |      |
| RECEIVED                                 |              |      |
| CLOSE                                    |              |      |
| FORECAST                                 |              |      |
| HOURLY_RATE                              |              |      |
| COST                                     |              |      |
| QUALITY_EVALUATION                       |              |      |
| MANUFACTURER                             |              |      |
| MANUFACTURER_SECTOR                      |              |      |
| MANUFACTURER_INDUSTRY_GROUP              |              |      |
| MANUFACTURER_INDUSTRY                    |              |      |
| MANUFACTURER_SUBINDUSTRY                 |              |      |
| NAME (from Schedules)                    |              |      |
| START (from Schedules)                   |              |      |
| END (from Schedules)                     |              |      |
| MON                                      |              |      |
| TUES                                     |              |      |
| WED                                      |              |      |
| THURS                                    |              |      |
| FRI                                      |              |      |
| SAT                                      |              |      |
| SUN                                      |              |      |
| CLIENT_NAME                              |              |      |
| SELLING_HOURLY_PRICE                     |              |      |
| MIN_QUALITY                              |              |      |
| WILDCARD                                 |              |      |
| TRANSLATOR (from TranslatorsCost+Pairs)  |              |      |
| SOURCE_LANG (from TranslatorsCost+Pairs) |              |      |
| TARGET_LANG (from TranslatorsCost+Pairs) |              |      |
| HOURLY_RATE (from TranslatorsCost+Pairs) |              |      |