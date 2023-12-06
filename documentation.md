# Data Slayer Thingy

# Thoughts

we need to clean:

> should_be_numerical = [

    "Fuel Consumption City (L/100 km)",
    "Fuel Consumption Hwy (L/100 km)",
    "Fuel Consumption Comb (L/100 km)",
    "Fuel Consumption Comb (mpg)",

]

Many unknown nan values → NaN, -1, "unspecified", "missing", "not-recorded", 9999, -9999, 0, etc

> PLEASE SEE THE DATA DESCRIPTION FROM THE KAGGLE TO CHECK.

#### What to do [[2023-12-07|Thursday]] 00:05 AM

1. QUICK EDA (PLOT, ETC) WITH MESSY, NOISY DATA
2. CLEAN ALL THE DATA (SHOULD BE NUMERICAL, NOT YET DROPPED) | [[2023-12-07|Thursday]] 01:39 AM

   1. change all the missing value (-1, 'unspecified', 'missing', …) to NaN value
   2. clean column 'should_be_numerical' [[2023-12-07|Thursday]] 00:24 AM (executed) | [[2023-12-07|Thursday]] 01:38 AM (done)

      1. do a regex to see all the pattern

         - THIS IS ALL THE AVAILABLE PATTERN:

           ```python

           ```

- Fuel Consumption City : [' liters per km' ' L/10km' ' km per L' ' L/100km' ' MPG (AS)' ' mpg Imp.' '-' ' L/ km' ' km/L' '' nan 'not-available' 'zero']
- Fuel Consumption Hwy : [' L/ km' ' liters per km' ' L/10km' ' mpg Imp.' ' L/100km' ' km per L' 'not-available' ' km/L' nan ' MPG (AS)' '-' '' 'zero']
- Fuel Consumption Comb : [' mpg Imp.' ' L/ km' 'zero' nan ' L/10km' ' MPG (AS)' ' km/L' ' L/100km' '-' ' km per L' ' liters per km' '' 'not-available']
  ```- SIDE NOTE:
  ![[Pasted image 20231207003613.png|200]] 2. analyze the pattern 3. for each pattern, to a standardization units to → liters per 100 km (source: [Bing Chat with GPT-4](https://www.bing.com/search?q=Bing+AI&showconv=1&FORM=undexpand)) 3. DROP/NP.NAN -1, -9999, 9999, etc
  > THIS IS THE RESULT AFTER CHANGING ALL WEIRD VALUE TO NAN
  > ![[Pasted image 20231207010841.png|300]]
  ```

3. CLEAN ALL THE DATA (CATEGORICAL) | [[2023-12-07|Thursday]] 02:13 AM
   - MISSING VALUE LIST FOR `VEHICLE CLASS`
     - VEHICLE CLASS
       - not-recorded
       - missing
       - na
       - not-available
       - unspecified
       - unestablished
       - unknown
       - -1
     - Transmission
       - not-recorded
       - unestablished
       - not-available
       - unspecified
       - unknown
       - -1
       - missing
     - Fuel Type
       - not-recorded
       - unspecified
       - unknown
       - missing
       - not-available
       - unestablished
       - na
       - -1
4. CLEAN ALL THE DATA (NUMERICAL) | [[2023-12-07|Thursday]] 02:21 AM
5. CHECK THE DATA WITH THE METADATA ON KAGGLE
6. Analyze the row of the missing value
7. Plot the data again
8. Do correlation
9. Drop outliers (maybe)
