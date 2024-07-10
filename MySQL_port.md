# Insights into Dog Behavior: Analyzing Dognition Data with MySQL
###### By: Zachary Raup


### Goal of Project:
The primary objective of this project is to utilize MySQL queries to conduct a comprehensive analysis of trends and correlations within the vast Dognition dataset. As part of the ‘Managing Big Data with MySQL’ course at Duke University, the project highlights improving skills in data cleaning, organizing, and applying advanced analytical techniques using SQL. By concentrating on these areas, the queries seek to reveal actionable insights from the dataset, deepening the understanding of dogs' behavioral patterns and preferences based on the collected data. This approach not only enables the practical use of database management techniques but also develops expertise in managing and interpreting large-scale datasets efficiently.



```python

```


### About the Data Set: Dognition

Dognition is a company that offers interactive tests designed for users to engage with their dogs. These tests are intended to create a personalized profile of the dog's personality based on collected data and information about the dog's background. The company uses this data to provide insights into various aspects of a dog's behavior and preferences.

The Dognition dataset encompasses a comprehensive range of information across six tables, featuring over 30 columns and exceeding 100,000 rows of data. This dataset includes details about users, such as their demographics and geographical information. It also captures specific attributes related to dogs, such as breed, age, and size. Additionally, the dataset covers information about the tests conducted and the corresponding results, offering a rich source of information for analyzing behavioral patterns and preferences in dogs.

By leveraging this dataset, Dognition aims to enhance understanding of canine behavior, helping dog owners discover activities and interests that their dogs may excel in or enjoy. This data-driven approach not only supports personalized recommendations but also contributes to broader insights into pet behavior and psychology.

###### * data set was not cleaned beforehand but was changed to protect personal information

[Link to Dognition Website](https://www.dognition.com/)


```python

```

### Writing Queries to Analyze the Data:



#### Test Completion Trends

##### Cleaning and Analyzing Data: Number of Tests Completed per Day of the Week
This query calculates the count of tests completed by distinct users with their dogs on each weekday across multiple years within the Dognition dataset. The dataset is filtered to include only users located in the contiguous United States (excluding Hawaii and Alaska) and excludes entries marked with "1" in their exclude columns to eliminate non-user testing inputs. The results are sorted first by year in ascending order and then by the total number of tests completed in descending order. The query utilizes a join function between the 'dogs' and 'users' tables to combine data from the complete_tests table (c) with a refined subset of dog and user data (dogs_cleaned). This ensures that the analysis focuses exclusively on tests associated with dogs owned by users meeting the specified geographical and exclusion criteria.




```sql
%%sql
SELECT DAYOFWEEK(c.created_at) AS dayasnum, YEAR(c.created_at) AS year,
COUNT(c.created_at) AS numtests,
(CASE
    WHEN DAYOFWEEK(c.created_at)=1 THEN "Su"
    WHEN DAYOFWEEK(c.created_at)=2 THEN "Mo"
    WHEN DAYOFWEEK(c.created_at)=3 THEN "Tu"
    WHEN DAYOFWEEK(c.created_at)=4 THEN "We"
    WHEN DAYOFWEEK(c.created_at)=5 THEN "Th"
    WHEN DAYOFWEEK(c.created_at)=6 THEN "Fr"
    WHEN DAYOFWEEK(c.created_at)=7 THEN "Sa"
END) AS daylabel
FROM complete_tests c JOIN
    (SELECT DISTINCT dog_guid
    FROM dogs d JOIN users u
        ON d.user_guid=u.user_guid
    WHERE ((u.exclude IS NULL OR u.exclude=0)
        AND u.country="US"
        AND (u.state!="HI" AND u.state!="AK")
        AND (d.exclude IS NULL OR d.exclude=0))) AS dogs_cleaned
    ON c.dog_guid=dogs_cleaned.dog_guid
GROUP BY year,daylabel
ORDER BY year ASC, numtests DESC
LIMIT 5;
```

     * mysql://studentuser:***@localhost/dognitiondb
    5 rows affected.
    




<table>
    <tr>
        <th>dayasnum</th>
        <th>year</th>
        <th>numtests</th>
        <th>daylabel</th>
    </tr>
    <tr>
        <td>1</td>
        <td>2013</td>
        <td>5860</td>
        <td>Su</td>
    </tr>
    <tr>
        <td>7</td>
        <td>2013</td>
        <td>4674</td>
        <td>Sa</td>
    </tr>
    <tr>
        <td>2</td>
        <td>2013</td>
        <td>3695</td>
        <td>Mo</td>
    </tr>
    <tr>
        <td>4</td>
        <td>2013</td>
        <td>3496</td>
        <td>We</td>
    </tr>
    <tr>
        <td>3</td>
        <td>2013</td>
        <td>3449</td>
        <td>Tu</td>
    </tr>
</table>




##### Actionable Insights:
Based on the SQL query results that analyze test completion trends within the Dognition dataset, several key insights emerge that can be used to optimize user engagement and test participation strategies:

###### Peak Test Completion Days:

    Sundays and Saturdays have the highest test completion rates, with 5,860 and 4,674 tests completed respectively in the year 2013. This indicates that users are more likely to engage in test activities during weekends.

    Recommendation: Increase marketing efforts and user engagement initiatives on weekends to leverage the higher user activity. Consider launching weekend-specific challenges or promotions to boost participation.

###### Weekday Trends:

    Among weekdays, Mondays show relatively higher engagement compared to the other weekdays in 2013.
    
    Recommendation: Schedule mid-week reminders or activities to maintain user engagement throughout the week. Introduce mid-week incentives or thematic content to sustain interest and participation.

By leveraging these insights, Dognition can optimize user engagement strategies to enhance user experience and drive higher participation rates in their testing activities. The regular analysis of test completion trends allows for targeted marketing efforts and timely reminders, ultimately fostering a more active and engaged community of users.


```python

```

#### Geographic User Participation Trends

##### Cleaning and Analyzing Data: Number of Distinct Users by State

This query calculates the number of distinct users who have completed tests with their dogs, segmented by state within the Dognition dataset. The dataset is filtered to include only users located in the contiguous United States (excluding Hawaii and Alaska) and excludes entries marked with "1" in their exclude columns to remove non-user inputs. The results are sorted by the total number of distinct users in descending order. The query joins the 'dogs' and 'users' tables to combine data from the complete_tests table (c) with a refined subset of dog and user data (dogs_cleaned). This ensures the analysis focuses exclusively on tests associated with dogs owned by users who meet the specified geographical and exclusion criteria.


```sql
%%sql
SELECT dogs_cleaned.state AS state, COUNT(DISTINCT dogs_cleaned.user_guid) AS
numusers
FROM complete_tests c JOIN
    (SELECT DISTINCT dog_guid, u.user_guid, u.state
    FROM dogs d JOIN users u
        ON d.user_guid=u.user_guid
    WHERE ((u.exclude IS NULL OR u.exclude=0)
        AND u.country="US"
        AND (d.exclude IS NULL OR d.exclude=0))) AS dogs_cleaned
    ON c.dog_guid=dogs_cleaned.dog_guid
GROUP BY state
ORDER BY numusers DESC
LIMIT 5;
```

     * mysql://studentuser:***@localhost/dognitiondb
    5 rows affected.
    




<table>
    <tr>
        <th>state</th>
        <th>numusers</th>
    </tr>
    <tr>
        <td>CA</td>
        <td>1363</td>
    </tr>
    <tr>
        <td>NY</td>
        <td>628</td>
    </tr>
    <tr>
        <td>TX</td>
        <td>536</td>
    </tr>
    <tr>
        <td>FL</td>
        <td>502</td>
    </tr>
    <tr>
        <td>NC</td>
        <td>467</td>
    </tr>
</table>



##### Actionable Insights:

The SQL query identifies the top 5 states with the highest number of Dognition users. This analysis uncovers several key insights that can be used to enhance user engagement and improve test participation strategies::

###### Peak User States:

    California (CA) has the highest number of distinct users, with 1,363 unique users engaging in test activities.New York (NY) follows with 628 users, Texas (TX) with 536 users, Florida (FL) with 502 users, and North Carolina (NC) with 467 users. The high participation rates in states like California, New York, and Texas suggest that these regions have a significant user base interested in Dognition activities.

    Recommendation: Focus marketing and engagement efforts on these top states to further capitalize on the existing user base. Consider organizing state-specific events or promotions to increase engagement in these high-participation areas. Tailor content and communication strategies to the preferences and interests of users in these states. Implement localized advertising and partnerships with regional organizations to strengthen community ties and drive further participation.


###### Underrepresented Regions:

    The query results provide a clear picture of regional engagement, allowing for targeted geographical analysis. This can be useful for identifying potential markets for expansion and understanding regional behavior patterns.

    Recommendation: Use these insights to identify underrepresented regions that may benefit from increased outreach efforts. Develop strategies to engage users in states with lower participation rates by addressing potential barriers and promoting the benefits of the Dognition program.

Utilizing these insights from the top 5 states with the highest number of Dognition users, Dognition can enhance user engagement strategies to improve user experience and drive higher participation rates in these key regions. Regular analysis of participation trends in these states allows for targeted marketing efforts, timely reminders, and region-specific initiatives, ultimately fostering a more active and engaged community of users in these high-participation areas.


```python

```



#### Test Completion Duration Trends

##### Cleaning and Analyzing Data: Average Test Complettion Time per Breed Type

This query calculates the average time it took for each dog breed type to complete all tests, along with the standard deviation, in the exam_answers table within the Dognition dataset. The analysis excludes any negative durations. The results are sorted by breed type and utilize a join between the 'dogs' and 'exam_answers' tables to combine the necessary data.


```sql
%%sql
SELECT d.breed_type AS breed_type,
AVG(TIMESTAMPDIFF(minute,e.start_time,e.end_time)) AS AvgDuration,
STDDEV(TIMESTAMPDIFF(minute,e.start_time,e.end_time)) AS StdDevDuration
FROM dogs d JOIN exam_answers e
    ON d.dog_guid=e.dog_guid
WHERE TIMESTAMPDIFF(minute,e.start_time,e.end_time)>0
GROUP BY breed_type;
```

     * mysql://studentuser:***@localhost/dognitiondb
    4 rows affected.
    




<table>
    <tr>
        <th>breed_type</th>
        <th>AvgDuration</th>
        <th>StdDevDuration</th>
    </tr>
    <tr>
        <td>Cross Breed</td>
        <td>11810.3230</td>
        <td>59113.45580229881</td>
    </tr>
    <tr>
        <td>Mixed Breed/ Other/ I Don&#x27;t Know</td>
        <td>9145.1575</td>
        <td>48748.626840777506</td>
    </tr>
    <tr>
        <td>Popular Hybrid</td>
        <td>7734.0763</td>
        <td>45577.65824281632</td>
    </tr>
    <tr>
        <td>Pure Breed</td>
        <td>12311.2558</td>
        <td>60997.35425304078</td>
    </tr>
</table>



##### Actionable Insights:

Based on the SQL query results analyzing the average duration and standard deviation of exam completion times for different breed types within the Dognition dataset, several key insights emerge that can inform user engagement and test optimization strategies:


###### Longest Test Completion Time:

    Pure breed dogs have the longest average exam duration of 12,311 minutes and the highest variability in completion times. Cross breed dogs exhibit the second highest average exam duration and significant variability.

    Recommendation: Investigate potential factors contributing to longer and more variable completion times for pure breed and cross breed dogs. Consider offering tailored guidance or support to owners of these breed types to streamline the testing process. Providing specific tips or resources could help these users complete exams more efficiently.


###### High Variability Between Completion Times:

    The query results show high variability, with each breed type exhibiting a high standard deviation in completion times.

    Recommendation: Further investigate the causes of high variability in test completion times. Determine whether this measure accurately reflects the pace at which different dog breeds complete the tests. Understanding the underlying reasons for this variability can inform the development of more effective engagement and support strategies.

Dognition can refine its user engagement strategies to enhance the testing experience across different breed types. Regular analysis of exam duration trends allows for targeted support and resources, helping to reduce variability and improve the overall efficiency of the testing process. This approach will ultimately lead to higher user satisfaction and increased participation rates in Dognition's testing activities.






```python

```


```python

```
###### Portfolio Links
[Zachary's Portfolio](README.md)  
[Project 1: Utilizing MCMC in Python to Explore the Parameter Space of an Exoplanet Transit](TOI4153_port.md)


```python

```


```python

```


```python

```


```python

```
