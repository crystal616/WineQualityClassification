# Wine Quality Analysis: Regression and Classification

# Data Description

Observations from 6497 varieties of red and white wine were collected from the Vinho Verde region of Portugal. Of those varieties, 1599 were red and 4898 were white. Data on the following 12 properties were collected - fixed acidity, volatile acidity, citric acid, residual sugar, chlorides, free sulfur dioxide, total sulfur dioxide, density, pH, sulphates, alcohol, and quality. The first 11 observations were quantitative chemical properties while the 12th observation was an ordinal score based on sensory data. This quality score was determined by taking the median of 3 independent testers' ratings on a scale from 1 (worst) to 10 (best).


# Purpose

The purpose of this analysis was to identify relationships between chemical properties of wine to the human quality of tasting in order to make the quality appraisal more objective and less reliant on practices dependent on human assessment.

# Conclusion

Our best regression method, forward subset selection, resulted in a test MSE of 0.419. The best classification method, random forests, gave a much smaller classification error rate of 0.159 (Table 2, p15, final report).
