import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

# ============ DATA PREPARATION ============

df = pd.read_csv('messy_breast_cancer.csv')
df.columns=(df.columns.str.strip().str.lower())
df.info()

df['age']=df['age'].str.replace('yrs|y|-','',regex=True)
df['age']=pd.to_numeric(df['age'],errors='coerce').astype('Int64')

df=df.rename(columns={'radius_mean':'radius_mean(mm)'})
df['radius_mean(mm)']=df['radius_mean(mm)'].str.strip().str.replace('mm|-','',regex=True)
df['radius_mean(mm)']=df['radius_mean(mm)'].str.replace(',','.')
df['radius_mean(mm)']=df['radius_mean(mm)'].astype('Float64')

df=df.drop_duplicates(subset='patient_id')
df=df.drop(columns='patient_id')

df['texture_mean']=df['texture_mean'].str.replace(',','.')
df['texture_mean']=df['texture_mean'].astype('Float64')

df['perimeter_mean']=df['perimeter_mean'].str.replace('mm','')
df['perimeter_mean']=df['perimeter_mean'].astype('Float64')

df['area (mm2)']=df['area (mm2)'].str.replace('mm²|mm2','',regex=True)
df['area (mm2)']=df['area (mm2)'].str.replace(',','.')
df['area (mm2)']=df['area (mm2)'].astype('Float64')
df=df.rename(columns={'area (mm2)':'area(mm2)'})

df['smoothness_mean']=df['smoothness_mean'].str.replace('%','')
df['smoothness_mean']=df['smoothness_mean'].astype('Float64')

df['family history']=(df['family history'].str.strip()
                      .str.lower()
                      .str.replace('yes|y|yess|true','1',regex=True)
                      .str.replace('no|n|false','0',regex=True)
                      .str.replace('1s','1')
                      .str.replace('0pe|0o','0',regex=True))

df['family history']=df['family history'].astype('Int64')
df=df.rename(columns={'family history':'family_history'})

df['concavity_mean']=df['concavity_mean'].str.replace(',','.')
df['concavity_mean']=df['concavity_mean'].astype('Float64')

df['menopause_status']=(df['menopause_status'].str.strip()
                        .str.lower()
                        .str.replace('-',''))
df['menopause_status']=(df['menopause_status'].str.replace('pre|premenopausal','0',regex=True)
                        .str.replace('0menopause|0menopausal','0',regex=True)
                        .str.replace('post|postmenopausal','1',regex=True)
                        .str.replace('1menopause|1menopausal','1',regex=True))
df['menopause_status']=df['menopause_status'].astype('Int64')

df['diagnosis']=(df['diagnosis'].str.strip()
                 .str.lower()
                 .str.replace('m|malignant','1',regex=True)
                 .str.replace('1alignant|101e1n1i1g1n1|1alignat|1aligant','1',regex=True)
                 .str.replace('b|benign','0',regex=True)
                 .str.replace('0enign|0engin','0',regex=True))
df['diagnosis']=df['diagnosis'].astype('Int64')

df.info()

# ============ TRAIN / TEST SPLIT ============

x=df.drop(columns='diagnosis')
y=df['diagnosis']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=22)

# ============ OUTLIER HANDLING ============
# Use IQR bounds calculated from the training set only to avoid data leakage.
# Clip extreme values instead of removing rows so the dataset stays intact.
numeric_columns = x_train.select_dtypes(include='number').columns
q1 = x_train[numeric_columns].quantile(0.25)
q3 = x_train[numeric_columns].quantile(0.75)
iqr = q3 - q1
lower_bounds = q1 - 1.5 * iqr
upper_bounds = q3 + 1.5 * iqr

x_train[numeric_columns] = x_train[numeric_columns].clip(
    lower=lower_bounds, upper=upper_bounds, axis=1
)
x_test[numeric_columns] = x_test[numeric_columns].clip(
    lower=lower_bounds, upper=upper_bounds, axis=1
)

# ============ FEATURE SCALING ============

scalar=StandardScaler()
x_train=scalar.fit_transform(x_train)
x_test=scalar.transform(x_test)

# ==== MODEL TRAINING / MODEL EVALUATION ====

model=LogisticRegression()
model.fit(x_train,y_train)
y_pred=model.predict(x_test)

cr=classification_report(y_test, y_pred, target_names=['Benign', 'Malignant'],output_dict=True)
cr_df = pd.DataFrame(cr).transpose().round(3)
cm=confusion_matrix(y_test,y_pred)

print("classification_report:\n")
print(cr_df)
print("Confusion Matrix:\n")
print(pd.DataFrame(cm,
                   index=['Actual Benign (0)', 'Actual Malignant (1)'],
                   columns=['Pred Benign (0)', 'Pred Malignant (1)']))

