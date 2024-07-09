# Insights into Dog Behavior: Analyzing Dognition Data with MySQL
###### By: Zachary Raup


### Goal of Project:
The goal of this project is to utilize MySQL queries to perform in-depth analysis of trends and relationships embedded within the Dognition database. Developed as a fundamental component of the ‘Managing Big Data with MySQL’ course at Duke University, the project focuses on refining and applying skills in data cleaning, sorting, and employing advanced analytical techniques using SQL. By exploring large datasets such as the Dognition database, the project aims to uncover meaningful insights into canine behavior patterns and preferences, leveraging robust data management practices to extract actionable intelligence for further research and practical applications in understanding and enhancing dog-human interactions.

```python

```

### About the Data Set: Dognition

Dognition is a company that offers interactive tests designed for users to engage with their dogs. These tests are intended to create a personalized profile of the dog's personality based on collected data and information about the dog's background. The company uses this data to provide insights into various aspects of a dog's behavior and preferences.

The Dognition dataset encompasses a comprehensive range of information across six tables, featuring over 30 columns and exceeding 100,000 rows of data. This dataset includes details about users, such as their demographics and geographical information. It also captures specific attributes related to dogs, such as breed, age, and size. Additionally, the dataset covers extensive information about the tests conducted and the corresponding results, offering a rich source of information for analyzing behavioral patterns and preferences in dogs.

By leveraging this dataset, Dognition aims to enhance understanding of canine behavior, helping dog owners discover activities and interests that their dogs may excel in or enjoy. This data-driven approach not only supports personalized recommendations but also contributes to broader insights into pet behavior and psychology.

[Link to Dognition Website](https://www.dognition.com/)


```python

```


### MySQL Queries to Analyze the Data:



#### 1. # of Tests Completed per Day of the Week

This query provides a count of the number of tests, distinct users completed with their dogs on each weekday of each year in the Dognition data set. 

##### Cleaning the data: 
Only users in the United States were to be shown but without hawaii or alaska. I excluded all dog_guids and user_guids with a value of "1" in their exclude columns to remove the companies example testing inputs that do not represent user data. The output is sorted by year in ascending order, and then by the total number of tests completed in descending order. I joined two tables (dogs and users) to focus on the distinct users that were represented in both tables. 



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




```python

```

#### Quesry cleans the data and lists the states by the most users 

Unfortunately other database platforms do not have the ORDER BY FIELD functionality.  To achieve the same result in other platforms, you would have to use a CASE statement or a more advanced solution:

http://stackoverflow.com/questions/1309624/simulating-mysqls-order-by-field-in-postgresql

The link provided above is to a discussion on stackoverflow.com.  Stackoverflow is a great website that, in their words, "is a community of 4.7 million programmers, just like you, helping each other."  You can ask questions about SQL queries and get help from other experts, or search through questions posted previously to see if somebody else has already asked a question that is relevant to the problem you are trying to solve.  It's a great resource to use whenever you run into trouble with your queries.

2. Which states and countries have the most Dognition users?

You ended up with a pretty long and complex query in the questions above that you tested step-by-step.  Many people save these types of queries so that they can be adapted for similar queries in the future without having to redesign and retest the entire query.  
    
In the next two questions, we will practice repurposing previously-designed queries for new questions.  Both questions can be answered through relatively minor modifications of the queries you wrote above.

**Question 14: Which 5 states within the United States have the most Dognition customers, once all dog_guids and user_guids with a value of "1" in their exclude columns are removed?  Try using the following general strategy: count how many unique user_guids are associated with dogs in the complete_tests table, break up the counts according to state, sort the results by counts of unique user_guids in descending order, and then limit your output to 5 rows. California ("CA") and New York ("NY") should be at the top of your list.**


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




```python

```


#### Write a query that calculates the average amount of time it took each dog breed_type to complete all of the tests in the exam_answers table. Exclude negative durations from the calculation, and include a column that calculates the standard deviation of durations for each breed_type group:



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




```python

```

### Actionable Insights from SQL Queries



```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```





Queries that Test Relationships Between Test Completion and Testing Circumstances 

In this lesson, we are going to practice integrating more of the concepts we learned over the past few weeks to address whether issues in our Dognition sPAP are related to the number of tests dogs complete.  We are going to focus on a subset of the issues listed in the "Features of Testing Circumstances" branch of our sPAP.  You will need to look up new functions several times and the final queries at which we will arrive by the end of this lesson will be quite complex, but we will work up to them step-by-step.  

To begin, load the sql library and database, and make the Dognition database your default database:


```python
%load_ext sql
%sql mysql://studentuser:studentpw@localhost/dognitiondb
%sql USE dognitiondb 
```


<img src="https://duke.box.com/shared/static/p2eucjdttai08eeo7davbpfgqi3zrew0.jpg" width=600 alt="SELECT FROM WHERE" />

## 1. During which weekdays do Dognition users complete the most tests?

The first question we are going to address is whether there is a certain day of the week when users are more or less likely to complete Dognition tests.  If so, targeting promotions or reminder emails to those times of the week might increase the number of tests users complete.

At first, the query we need to address this question might seem a bit intimidating, but once you can describe what the query needs to do in words, writing the query won't seem so challenging.  

Ultimately, we want a count of the number of tests completed on each day of the week, with all of the dog_guids and user_guids the Dognition team flagged in their exclude column excluded.  To achieve this, we are going to have to use the GROUP BY clause to break up counts of the records in the completed_tests table according to days of the week.  We will also have to join the completed_tests table with the dogs and users table in order to exclude completed_tests records that are associated with dog_guids or user_guids that should be excluded.  First, though, we need a method for extracting the day of the week from a time stamp.  In MySQL Exercise 2 we used a function called "DAYNAME".  That is the most efficient function to use for this purpose, but not all database systems have this function, so let's try using a different method for the queries in this lesson.   Search these sites to find a function that will output a number from 1-7 for time stamps where 1 = Sunday, 2 = Monday, …, 7 = Saturday:

https://dev.mysql.com/doc/refman/5.7/en/sql-function-reference.html
http://www.w3resource.com/mysql/mysql-functions-and-operators.php

**Question 1: Using the function you found in the websites above, write a query that will output one column with the original created_at time stamp from each row in the completed_tests table, and another column with a number that represents the day of the week associated with each of those time stamps.  Limit your output to 200 rows starting at row 50.**


```sql
%%sql
SELECT created_at, DAYOFWEEK(created_at)
FROM complete_tests
LIMIT 49,5;
```

     * mysql://studentuser:***@localhost/dognitiondb
    5 rows affected.
    




<table>
    <tr>
        <th>created_at</th>
        <th>DAYOFWEEK(created_at)</th>
    </tr>
    <tr>
        <td>2013-02-05 22:10:06</td>
        <td>3</td>
    </tr>
    <tr>
        <td>2013-02-05 22:23:49</td>
        <td>3</td>
    </tr>
    <tr>
        <td>2013-02-05 22:26:36</td>
        <td>3</td>
    </tr>
    <tr>
        <td>2013-02-05 22:29:02</td>
        <td>3</td>
    </tr>
    <tr>
        <td>2013-02-05 22:32:25</td>
        <td>3</td>
    </tr>
</table>



Of course, the results of the query in Question 1 would be much easier to interpret if the output included the name of the day of the week (or a relevant abbreviation) associated with each time stamp rather than a number index.

**Question 2: Include a CASE statement in the query you wrote in Question 1 to output a third column that provides the weekday name (or an appropriate abbreviation) associated with each created_at time stamp.**


```sql
%%sql
SELECT created_at, DAYOFWEEK(created_at),
    (CASE
    WHEN DAYOFWEEK(created_at)=1 THEN "Su"
    WHEN DAYOFWEEK(created_at)=2 THEN "Mo"
    WHEN DAYOFWEEK(created_at)=3 THEN "Tu"
    WHEN DAYOFWEEK(created_at)=4 THEN "We"
    WHEN DAYOFWEEK(created_at)=5 THEN "Th"
    WHEN DAYOFWEEK(created_at)=6 THEN "Fr"
    WHEN DAYOFWEEK(created_at)=7 THEN "Sa"
    END) AS daylabel
FROM complete_tests
LIMIT 49,5;
```

     * mysql://studentuser:***@localhost/dognitiondb
    5 rows affected.
    




<table>
    <tr>
        <th>created_at</th>
        <th>DAYOFWEEK(created_at)</th>
        <th>daylabel</th>
    </tr>
    <tr>
        <td>2013-02-05 22:10:06</td>
        <td>3</td>
        <td>Tu</td>
    </tr>
    <tr>
        <td>2013-02-05 22:23:49</td>
        <td>3</td>
        <td>Tu</td>
    </tr>
    <tr>
        <td>2013-02-05 22:26:36</td>
        <td>3</td>
        <td>Tu</td>
    </tr>
    <tr>
        <td>2013-02-05 22:29:02</td>
        <td>3</td>
        <td>Tu</td>
    </tr>
    <tr>
        <td>2013-02-05 22:32:25</td>
        <td>3</td>
        <td>Tu</td>
    </tr>
</table>




```sql
%%sql
SELECT DAYOFWEEK(DATE_SUB(created_at, interval 6 hour)) AS dayasnum,
YEAR(c.created_at) AS year, COUNT(c.created_at) AS numtests,
    (CASE
    WHEN DAYOFWEEK(DATE_SUB(created_at, interval 6 hour))=1 THEN "Su"
    WHEN DAYOFWEEK(DATE_SUB(created_at, interval 6 hour))=2 THEN "Mo"
    WHEN DAYOFWEEK(DATE_SUB(created_at, interval 6 hour))=3 THEN "Tu"
    WHEN DAYOFWEEK(DATE_SUB(created_at, interval 6 hour))=4 THEN "We"
    WHEN DAYOFWEEK(DATE_SUB(created_at, interval 6 hour))=5 THEN "Th"
    WHEN DAYOFWEEK(DATE_SUB(created_at, interval 6 hour))=6 THEN "Fr"
    WHEN DAYOFWEEK(DATE_SUB(created_at, interval 6 hour))=7 THEN "Sa"
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
ORDER BY year ASC, FIELD(daylabel,'Mo','Tu','We','Th','Fr','Sa','Su')
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
        <td>2</td>
        <td>2013</td>
        <td>3798</td>
        <td>Mo</td>
    </tr>
    <tr>
        <td>3</td>
        <td>2013</td>
        <td>3276</td>
        <td>Tu</td>
    </tr>
    <tr>
        <td>4</td>
        <td>2013</td>
        <td>3410</td>
        <td>We</td>
    </tr>
    <tr>
        <td>5</td>
        <td>2013</td>
        <td>3079</td>
        <td>Th</td>
    </tr>
    <tr>
        <td>6</td>
        <td>2013</td>
        <td>3049</td>
        <td>Fr</td>
    </tr>
</table>



Now that we are confident we have the correct syntax for extracting weekday labels from the created_at time stamps, we can start building our larger query that examines the number of tests completed on each weekday.

**Question 3: Adapt the query you wrote in Question 2 to report the total number of tests completed on each weekday.  Sort the results by the total number of tests completed in descending order.  You should get a total of 33,190 tests in the Sunday row of your output.**


```sql
%%sql
SELECT DAYOFWEEK(created_at),COUNT(created_at) AS numtests,
    (CASE
    WHEN DAYOFWEEK(created_at)=1 THEN "Su"
    WHEN DAYOFWEEK(created_at)=2 THEN "Mo"
    WHEN DAYOFWEEK(created_at)=3 THEN "Tu"
    WHEN DAYOFWEEK(created_at)=4 THEN "We"
    WHEN DAYOFWEEK(created_at)=5 THEN "Th"
    WHEN DAYOFWEEK(created_at)=6 THEN "Fr"
    WHEN DAYOFWEEK(created_at)=7 THEN "Sa"
    END) AS daylabel
FROM complete_tests
GROUP BY daylabel
ORDER BY numtests DESC;
```

     * mysql://studentuser:***@localhost/dognitiondb
    7 rows affected.
    




<table>
    <tr>
        <th>DAYOFWEEK(created_at)</th>
        <th>numtests</th>
        <th>daylabel</th>
    </tr>
    <tr>
        <td>1</td>
        <td>33190</td>
        <td>Su</td>
    </tr>
    <tr>
        <td>2</td>
        <td>30195</td>
        <td>Mo</td>
    </tr>
    <tr>
        <td>3</td>
        <td>27989</td>
        <td>Tu</td>
    </tr>
    <tr>
        <td>7</td>
        <td>27899</td>
        <td>Sa</td>
    </tr>
    <tr>
        <td>4</td>
        <td>26473</td>
        <td>We</td>
    </tr>
    <tr>
        <td>5</td>
        <td>24420</td>
        <td>Th</td>
    </tr>
    <tr>
        <td>6</td>
        <td>23080</td>
        <td>Fr</td>
    </tr>
</table>



So far these results suggest that users complete the most tests on Sunday night and the fewest tests on Friday night.  We need to determine if this trend remains after flagged dog_guids and user_guids are excluded.  Let's start by removing the dog_guids that have an exclude flag.  We'll exclude user_guids with an exclude flag in later queries.

**Question 4: Rewrite the query in Question 3 to exclude the dog_guids that have a value of "1" in the exclude column (Hint: this query will require a join.)  This time you should get a total of 31,092 tests in the Sunday row of your output.**

You can try re-running the query with time-zone corrections of 5, 7, or 8 hours, and the results remain essentially the same.  All of these analyses suggest that customers are most likely to complete tests around Sunday and Monday, and least likely to complete tests around the end of the work week, on Thursday and Friday. This is certainly valuable information for Dognition to take advantage of.

If you were presenting this information to the Dognition team, you might want to present the information in the form of a graph that you make in another program.  The graph would be easier to read if the output was ordered according to the days of the week shown in standard calendars, with Monday being the first day and Sunday being the last day.  MySQL provides an easy way to do this using the FIELD function in the ORDER BY statement:

https://www.virendrachandak.com/techtalk/mysql-ordering-results-by-specific-field-values/

**Question 13: Adapt your query from Question 12 so that the results are sorted by year in ascending order, and then by the day of the week in the following order: Monday, Tuesday, Wednesday, Thursday, Friday, Saturday, Sunday.**


```sql
%%sql
SELECT DAYOFWEEK(c.created_at),COUNT(c.created_at) AS numtests,
    (CASE
    WHEN DAYOFWEEK(c.created_at)=1 THEN "Su"
    WHEN DAYOFWEEK(c.created_at)=2 THEN "Mo"
    WHEN DAYOFWEEK(c.created_at)=3 THEN "Tu"
    WHEN DAYOFWEEK(c.created_at)=4 THEN "We"
    WHEN DAYOFWEEK(c.created_at)=5 THEN "Th"
    WHEN DAYOFWEEK(c.created_at)=6 THEN "Fr"
    WHEN DAYOFWEEK(c.created_at)=7 THEN "Sa"
    END) AS daylabel
FROM complete_tests c JOIN dogs d
    ON c.dog_guid=d.dog_guid
WHERE d.exclude IS NULL OR d.exclude=0
GROUP BY daylabel
ORDER BY numtests DESC;
```

     * mysql://studentuser:***@localhost/dognitiondb
    7 rows affected.
    




<table>
    <tr>
        <th>DAYOFWEEK(c.created_at)</th>
        <th>numtests</th>
        <th>daylabel</th>
    </tr>
    <tr>
        <td>1</td>
        <td>31092</td>
        <td>Su</td>
    </tr>
    <tr>
        <td>2</td>
        <td>28250</td>
        <td>Mo</td>
    </tr>
    <tr>
        <td>7</td>
        <td>26231</td>
        <td>Sa</td>
    </tr>
    <tr>
        <td>3</td>
        <td>25764</td>
        <td>Tu</td>
    </tr>
    <tr>
        <td>4</td>
        <td>24501</td>
        <td>We</td>
    </tr>
    <tr>
        <td>5</td>
        <td>22347</td>
        <td>Th</td>
    </tr>
    <tr>
        <td>6</td>
        <td>21028</td>
        <td>Fr</td>
    </tr>
</table>



Now we need to exclude the user_guids that have a value of "1" in the exclude column as well.  One way to do this would be to join the completed_tests, dogs, and users table with a sequence of inner joins.  However, we've seen in previous lessons that there are duplicate rows in the users table.  These duplicates will get passed through the join and will affect the count calculations.  To illustrate this, compare the following two queries.

**Question 5: Write a query to retrieve all the dog_guids for users common to the dogs and users table using the traditional inner join syntax (your output will have 950,331 rows).**


```sql
%%sql
SELECT dog_guid
FROM dogs d INNER JOIN users u
    ON d.user_guid=u.user_guid
LIMIT 5;
```

     * mysql://studentuser:***@localhost/dognitiondb
    5 rows affected.
    




<table>
    <tr>
        <th>dog_guid</th>
    </tr>
    <tr>
        <td>fd27b272-7144-11e5-ba71-058fbc01cf0b</td>
    </tr>
    <tr>
        <td>fd417cac-7144-11e5-ba71-058fbc01cf0b</td>
    </tr>
    <tr>
        <td>fd27b5ba-7144-11e5-ba71-058fbc01cf0b</td>
    </tr>
    <tr>
        <td>fd3fb0f2-7144-11e5-ba71-058fbc01cf0b</td>
    </tr>
    <tr>
        <td>fd27b6b4-7144-11e5-ba71-058fbc01cf0b</td>
    </tr>
</table>



**Question 6: Write a query to retrieve all the *distinct* dog_guids common to the dogs and users table using the traditional inner join syntax (your output will have 35,048 rows).**


```sql
%%sql
SELECT DISTINCT dog_guid
FROM dogs d JOIN users u
ON d.user_guid=u.user_guid
LIMIT 5;
```

     * mysql://studentuser:***@localhost/dognitiondb
    5 rows affected.
    




<table>
    <tr>
        <th>dog_guid</th>
    </tr>
    <tr>
        <td>fd27b272-7144-11e5-ba71-058fbc01cf0b</td>
    </tr>
    <tr>
        <td>fd417cac-7144-11e5-ba71-058fbc01cf0b</td>
    </tr>
    <tr>
        <td>fd27b5ba-7144-11e5-ba71-058fbc01cf0b</td>
    </tr>
    <tr>
        <td>fd3fb0f2-7144-11e5-ba71-058fbc01cf0b</td>
    </tr>
    <tr>
        <td>fd27b6b4-7144-11e5-ba71-058fbc01cf0b</td>
    </tr>
</table>



The strategy we will use to handle duplicate rows in the users table will be to, first, write a subquery that retrieves the distinct dog_guids from an inner join between the dogs and users table with the appropriate records excluded.  Then, second, we will join the result of this subquery to the complete_tests table and group the results according to the day of the week.

**Question 7: Start by writing a query that retrieves distinct dog_guids common to the dogs and users table, excuding dog_guids and user_guids with a "1" in their respective exclude columns (your output will have 34,121 rows).**


```sql
%%sql
SELECT DISTINCT dog_guid
FROM dogs d JOIN users u
ON d.user_guid=u.user_guid
WHERE (u.exclude IS NULL OR u.exclude=0) AND (d.exclude IS NULL OR
d.exclude=0)
LIMIT 5;
```

     * mysql://studentuser:***@localhost/dognitiondb
    5 rows affected.
    




<table>
    <tr>
        <th>dog_guid</th>
    </tr>
    <tr>
        <td>fd27b272-7144-11e5-ba71-058fbc01cf0b</td>
    </tr>
    <tr>
        <td>fd417cac-7144-11e5-ba71-058fbc01cf0b</td>
    </tr>
    <tr>
        <td>fd27b5ba-7144-11e5-ba71-058fbc01cf0b</td>
    </tr>
    <tr>
        <td>fd3fb0f2-7144-11e5-ba71-058fbc01cf0b</td>
    </tr>
    <tr>
        <td>fd27b6b4-7144-11e5-ba71-058fbc01cf0b</td>
    </tr>
</table>



**Question 8: Now adapt your query from Question 4 so that it inner joins on the result of the subquery you wrote in Question 7 instead of the dogs table.  This will give you a count of the number of tests completed on each day of the week, excluding all of the dog_guids and user_guids that the Dognition team flagged in the exclude columns.**  


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
        AND (d.exclude IS NULL OR d.exclude=0))) AS dogs_cleaned
    ON c.dog_guid=dogs_cleaned.dog_guid
GROUP BY daylabel
ORDER BY numtests DESC;
```

     * mysql://studentuser:***@localhost/dognitiondb
    7 rows affected.
    




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
        <td>31036</td>
        <td>Su</td>
    </tr>
    <tr>
        <td>2</td>
        <td>2013</td>
        <td>28138</td>
        <td>Mo</td>
    </tr>
    <tr>
        <td>7</td>
        <td>2013</td>
        <td>26149</td>
        <td>Sa</td>
    </tr>
    <tr>
        <td>3</td>
        <td>2013</td>
        <td>25696</td>
        <td>Tu</td>
    </tr>
    <tr>
        <td>4</td>
        <td>2013</td>
        <td>24433</td>
        <td>We</td>
    </tr>
    <tr>
        <td>5</td>
        <td>2014</td>
        <td>22323</td>
        <td>Th</td>
    </tr>
    <tr>
        <td>6</td>
        <td>2013</td>
        <td>21027</td>
        <td>Fr</td>
    </tr>
</table>



These results still suggest that Sunday is the day when the most tests are completed and Friday is the day when the fewest tests are completed.  However, our first query suggested that more tests were completed on Tuesday than Saturday; our current query suggests that slightly more tests are completed on Saturday than Tuesday, now that flagged dog_guids and user_guids are excluded.

It's always a good idea to see if a data pattern replicates before you interpret it too strongly.  The ideal way to do this would be to have a completely separate and independent data set to analyze.  We don't have such a data set, but we can assess the reliability of the day of the week patterns in a different way.  We can test whether the day of the week patterns are the same in all years of our data set.

**Question 9: Adapt your query from Question 8 to provide a count of the number of tests completed on each weekday of each year in the Dognition data set.  Exclude all dog_guids and user_guids with a value of "1" in their exclude columns.  Sort the output by year in ascending order, and then by the total number of tests completed in descending order. HINT: you will need a function described in one of these references to retrieve the year of each time stamp in the created_at field:**

https://dev.mysql.com/doc/refman/5.7/en/sql-function-reference.html
http://www.w3resource.com/mysql/mysql-functions-and-operators.php


```sql
%%sql SELECT DAYOFWEEK(c.created_at) AS dayasnum, YEAR(c.created_at) AS
year, COUNT(c.created_at) AS numtests,
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
        <td>8203</td>
        <td>Su</td>
    </tr>
    <tr>
        <td>7</td>
        <td>2013</td>
        <td>6854</td>
        <td>Sa</td>
    </tr>
    <tr>
        <td>2</td>
        <td>2013</td>
        <td>5740</td>
        <td>Mo</td>
    </tr>
    <tr>
        <td>4</td>
        <td>2013</td>
        <td>5665</td>
        <td>We</td>
    </tr>
    <tr>
        <td>3</td>
        <td>2013</td>
        <td>5393</td>
        <td>Tu</td>
    </tr>
</table>



The next step is to adjust the created_at times for differences in time zone. Most United States states (excluding Hawaii and Alaska) have a time zone of UTC time -5 hours (in the eastern-most regions) to -8 hours (in the western-most regions).  To get a general idea for how much our weekday analysis is likely to change based on time zone, we will subtract 6 hours from every time stamp in the complete_tests table.  Although this means our time stamps can be inaccurate by 1 or 2 hours, people are not likely to be playing Dognition games at midnight, so 1-2 hours should not affect the weekdays extracted from each time stamp too much. 

The functions used to subtract time differ across database systems, so you should double-check which function you need to use every time you are working with a new database.  We will use the date_sub function:

https://www.w3schools.com/sql/func_mysql_date_sub.asp

**Question 11: Write a query that extracts the original created_at time stamps for rows in the complete_tests table in one column, and the created_at time stamps with 6 hours subtracted in another column.  Limit your output to 100 rows.**


```sql
%%sql
SELECT created_at, DATE_SUB(created_at, interval 6 hour) AS corrected_time
FROM complete_tests
LIMIT 5;
```

     * mysql://studentuser:***@localhost/dognitiondb
    5 rows affected.
    




<table>
    <tr>
        <th>created_at</th>
        <th>corrected_time</th>
    </tr>
    <tr>
        <td>2013-02-05 18:26:54</td>
        <td>2013-02-05 12:26:54</td>
    </tr>
    <tr>
        <td>2013-02-05 18:31:03</td>
        <td>2013-02-05 12:31:03</td>
    </tr>
    <tr>
        <td>2013-02-05 18:32:04</td>
        <td>2013-02-05 12:32:04</td>
    </tr>
    <tr>
        <td>2013-02-05 18:32:25</td>
        <td>2013-02-05 12:32:25</td>
    </tr>
    <tr>
        <td>2013-02-05 18:32:56</td>
        <td>2013-02-05 12:32:56</td>
    </tr>
</table>



**Question 12: Use your query from Question 11 to adapt your query from Question 10 in order to provide a count of the number of tests completed on each day of the week, with approximate time zones taken into account, in each year in the Dognition data set. Exclude all dog_guids and user_guids with a value of "1" in their exclude columns. Sort the output by year in ascending order, and then by the total number of tests completed in descending order. HINT: Don't forget to adjust for the time zone in your DAYOFWEEK statement and your CASE statement.** 


```sql
%%sql
SELECT DAYOFWEEK(DATE_SUB(created_at, interval 6 hour)) AS dayasnum,
YEAR(c.created_at) AS year, COUNT(c.created_at) AS numtests,
    (CASE
    WHEN DAYOFWEEK(DATE_SUB(created_at, interval 6 hour))=1 THEN "Su"
    WHEN DAYOFWEEK(DATE_SUB(created_at, interval 6 hour))=2 THEN "Mo"
    WHEN DAYOFWEEK(DATE_SUB(created_at, interval 6 hour))=3 THEN "Tu"
    WHEN DAYOFWEEK(DATE_SUB(created_at, interval 6 hour))=4 THEN "We"
    WHEN DAYOFWEEK(DATE_SUB(created_at, interval 6 hour))=5 THEN "Th"
    WHEN DAYOFWEEK(DATE_SUB(created_at, interval 6 hour))=6 THEN "Fr"
    WHEN DAYOFWEEK(DATE_SUB(created_at, interval 6 hour))=7 THEN "Sa"
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
        <td>6061</td>
        <td>Su</td>
    </tr>
    <tr>
        <td>7</td>
        <td>2013</td>
        <td>4754</td>
        <td>Sa</td>
    </tr>
    <tr>
        <td>2</td>
        <td>2013</td>
        <td>3798</td>
        <td>Mo</td>
    </tr>
    <tr>
        <td>4</td>
        <td>2013</td>
        <td>3410</td>
        <td>We</td>
    </tr>
    <tr>
        <td>3</td>
        <td>2013</td>
        <td>3276</td>
        <td>Tu</td>
    </tr>
</table>



The number of unique Dognition users in California is more than two times greater than any other state.  This information could be very helpful to Dognition.  Useful follow-up questions would be: were special promotions run in California that weren't run in other states?  Did Dognition use advertising channels that are particularly effective in California?  If not, what traits differentiate California users from other users?  Can these traits be taken advantage of in future marketing efforts or product developments?

Let's try one more analysis that examines testing circumstances from a different angle.

**Question 1: Which 10 countries have the most Dognition customers, once all dog_guids and user_guids with a value of "1" in their exclude columns are removed? HINT: don't forget to remove the u.country="US" statement from your WHERE clause.**


```sql
%%sql
SELECT dogs_cleaned.country AS country, COUNT(DISTINCT
dogs_cleaned.user_guid) AS numusers
FROM complete_tests c JOIN
(SELECT DISTINCT dog_guid, u.user_guid, u.country
FROM dogs d JOIN users u
ON d.user_guid=u.user_guid
WHERE ((u.exclude IS NULL OR u.exclude=0)
AND (d.exclude IS NULL OR d.exclude=0))) AS dogs_cleaned
ON c.dog_guid=dogs_cleaned.dog_guid
GROUP BY country
ORDER BY numusers DESC
LIMIT 10;
```

     * mysql://studentuser:***@localhost/dognitiondb
    

The United States, Canada, Australia, and Great Britain are the countries with the most Dognition users.  N/A refers to "not applicable" which essentially means we have no usable country data from those rows.  After Great Britain, the number of Dognition users drops quite a lot.  This analysis suggests that Dognition is most likely to be used by English-speaking countries.  One question Dognition might want to consider is whether there are any countries whose participation would dramatically increase if a translated website were available.

## 3. Congratulations!

You have now written many complex queries on your own that address real analysis questions about a real business problem.  You know how to look up new functions, you know how to troubleshoot your queries by isolating each piece of the query until you are sure the syntax is correct, and you know where to look for help if you get stuck.  You are ready to start using SQL in your own business ventures.  Keep learning, keep trying new things, and keep asking questions.  Congratulations for taking your career to the next level!

There is another video to watch, and of course, more exercises to work through using the Dillard's data set.  
    
**In the meantime, enjoy practicing any other queries you want to try here:**


```python

```
