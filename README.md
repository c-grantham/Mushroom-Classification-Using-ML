<h1>Introduction</h1>

The problem I am addressing with this project is classifying between edible and
poisonous mushrooms. Accurately and efficiently identifying potentially toxic mushrooms is
very important, as ingesting some poisonous species can be fatal within days if untreated.
Additionally, many highly toxic fungi closely resemble edible species, increasing the need for
adept classification. To address this problem, I generated a machine learning model that can
differentiate between edible and poisonous mushrooms trained on a publicly available dataset.

<h2>Dataset Overview</h2>
The dataset in question is the “Mushroom Classification” dataset, sourced from the UCI
Machine Learning Repository. It contains 8,124 hypothetical mushroom samples with 21
features, including edibility, odor, population, habitat, and characteristics of the cap, stalk, and
gills. These features provide a diverse range of examinable traits that help to distinguish between
different species of mushrooms, making this dataset ideal for machine learning modeling.

<h3>Social Impacts</h3>
The dataset’s comprehensive collection of physical and environmental information about
mushrooms has significant real-world implications, particularly in regards to public safety and
culinary usage. As previously mentioned, identifying poisonous and edible mushrooms
accurately is crucial, since consuming toxic mushrooms can lead to severe illness and even
death. Therefore, utilizing this dataset to train a machine learning model can provide invaluable
identification services to mycologists, foragers, and others interested in mushrooms.

From a societal perspective, this dataset can improve public health and safety. Many
mushrooms resemble each other, with some poisonous species appearing nearly identical to their
edible counterparts. One of the world’s deadliest mushrooms is the suitably named the Death
Cap, which closely resembles edible straw mushrooms and caesar's mushrooms. Creating a
method for accurate identification can guide safety initiatives on a national scale. It can also
provide crucial information for first responders in mushroom consumption related emergencies.
This dataset has broad societal impacts that has the potential to increase public safety, strengthen
knowledge about mushrooms, and reduce cases of misidentified toxic mushrooms.

<h4>Motivations</h4>
From a personal perspective, this dataset has crucial information for those who forage for
mushrooms, whether for culinary usage or as a hobby. These individuals often rely solely on
visual characteristics they perceive to distinguish between safe and unsafe mushrooms. The
ability to accurately classify mushrooms is a matter of personal safety, requiring individuals to
risk their health if they are not knowledgeable or experienced enough. A classification model
would eliminate this uncertainty, allowing foragers to enjoy their hobby without wagering their
well-being.

I was motivated to select the mycology domain because I wanted to learn more about
mushrooms and foraging in general. Personally, I think mushrooms are quite interesting so when
I came across this dataset I knew I had to choose it for this project. Knowing that there are
foragers out there who could potentially be risking their health and safety unknowingly also
motivated me to choose this dataset. If I have the ability to create a predictive model that would
eliminate these risk factors then I feel that I am required to do so.
The first step I took to prepare the data was to import it into the code file using a function
from the pandas library. Next, I separated the data into the “X” (independent variables) and “Y”
(dependent variable). The dependent variable is the first column “class” feature, which classifies
the mushrooms as either poisonous (p) or edible (e). This variable is the dependent variable
because it is the variable that I am trying to predict, it is also referred to as the target variable.

<h5>Data Preparation</h5>
Before any other data preprocessing is done, the missing values within the dataset need to be
filled in. I used the SimpleImputer with the “most_frequent” strategy, which fills in the empty
data points with the most frequently occurring value in that column. Once that is complete, the
OneHotEncoder is used along with the ColumnTransformer to encode and apply those
transformations to the dataset. I chose to use the OneHotEncoder instead of the LabelEncoder or
another encoding method because it converts the categorical variables into a unique binary
sequence for each category. This is preferential to the LabelEncoder’s method because it avoids
implying an ordinal relationship between the categories. The LabelEncoder has the potential to
do so since it converts the categories into numerical values. Then the dataset can be split into the
training set and test set, I chose to split the dataset 70/30 to allow for ample test data. Next, I
scaled the training and test set of independent variables, the dependent variable does not need to
be scaled since it is just a binary variable. I chose to use the StandardScaler instead of the
Normalizer because it is suited for datasets whose features have varying units and ranges.
However, I encountered an error with the StandardScaler being unable to center the data by
subtracting the mean because the data is too sparse. After investigating, I reasoned that the
compiler is throwing this error regarding sparse data likely because many of the elements are
zero since they were encoded with the OneHotEncoder. To correct this, the error recommended
passing the StandardScaler with “with_mean=False” to avoid attempting to center a sparse
matrix of data. Supplementing the scaler with this code snippet remedied the issue. These are all
of the steps I took to prepare the mushroom dataset for modeling.

<h6>Model Selection</h6>
When first considering what model would be best to use, I investigated what uses the
potential models were best suited for. Knowing that my dataset is high-dimensional and complex
I assumed a linear model would likely not perform well, but I still wanted to give one a try just
for some comparison to the non-linear models. For this I chose to consider Logistic Regression
as one of my models, since it is primarily used for binary classification. A Decision Tree model
was also added to my consideration because it is useful for complex classification problems.
Naturally, I also decided to investigate Random Forest as a potential model since it is an
ensemble of decision trees ideal for handling classification of high-dimensional data.
Additionally, I chose to analyze K-Nearest Neighbors and a non-linear Support Vector Machine,
specifically with the Radial Basis Function (RBF) kernel since they are capable of separating
non-linear classes. Finally, I wanted to consider a neural network as well on account of their
ability to identify complex non-linear relationships.

<h7>Chosen Model</h7>
Ultimately, I chose to use Random Forests to model the mushroom dataset because it
offers an advantage that addresses a key challenge associated with handling high-dimensional
data and complex relationships amongst classes—overfitting. Random Forest uses ensemble
learning techniques that build multiple decision trees with different subsets of data and combines
their predictions to make classifications. This method not only reduces critical overfitting risks of
only a single decision tree but also increases the robustness of the model.

<h8>Model Optimization</h8>
After a few initial builds of the model, I recorded extremely high accuracy of 99.9%. The
confusion matrix reported only one false positive and one false negative. I immediately assumed
that overfitting had occurred, and researched what I could do to prevent and identify overfitting
in future iterations. To identify overfitting in my model more concretely, I implemented cross
validation scores using cross_val_score from the model_selection library of sklearn. Cross
validation (CV) scores are a metric used to measure how well the model generalizes on new data.
Therefore, if the cross validation scores are lower than the regular accuracy of the model,
overfitting likely occurred. After reiterating the model with CV scores, I found that this was
definitely the case with my model, since the CV scores were over 30% lower than my 99%
accuracy. Since I had identified overfitting in my model, I now had to correct it. One technique I
learned from lecture was to adjust the ratio of test and training data, so I changed that to 70/30
from 80/20. This change did not significantly improve the CV scores, so I considered other
techniques. I learned about GridSearchCV, which also belongs to the model_selection library of
sklearn. GridSearchCV is a technique for finding the best hyperparameters for a model by
specifying a grid of parameters and evaluating the model’s performance for each combination of
those parameters. Utilizing this technique I determined the optimal parameters for accuracy and
minimal to no overfitting. After implementing these hyperparameters into the model, the
standard accuracy of the model was 99.7% and the average CV score was also 99.7%, indicating
that the overfitting issue had been resolved. Overall, the classification of mushrooms is
extremely accurate with 41 false positives and false negatives in comparison to 12,173 true
positives and true negatives.

<h9>Citations</h9>

[1] “7 of the World’s Most Poisonous Mushrooms | Britannica,” www.britannica.com.
https://www.britannica.com/story/7-of-the-worlds-most-poisonous-mushrooms
[2] SciKit Learn, “sklearn.model_selection.GridSearchCV — scikit-learn 0.22
Documentation,” Scikit-learn.org, 2019.
https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html
[3] “sklearn.model_selection.cross_val_score — scikit-learn 0.22 documentation,”
Scikit-learn.org, 2019.
https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_val_score.html
