{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "3721e86d",
   "metadata": {
    "_cell_guid": "b1076dfc-b9ad-4769-8c92-a6c4dae69d19",
    "_uuid": "8f2839f25d086af736a60e9eeb907d3b93b6e0e5",
    "execution": {
     "iopub.execute_input": "2022-12-06T21:41:12.498117Z",
     "iopub.status.busy": "2022-12-06T21:41:12.497373Z",
     "iopub.status.idle": "2022-12-06T21:41:12.510407Z",
     "shell.execute_reply": "2022-12-06T21:41:12.508707Z"
    },
    "papermill": {
     "duration": 0.020943,
     "end_time": "2022-12-06T21:41:12.512901",
     "exception": false,
     "start_time": "2022-12-06T21:41:12.491958",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "/kaggle/input/titanic/train.csv\n",
      "/kaggle/input/titanic/test.csv\n",
      "/kaggle/input/titanic/gender_submission.csv\n"
     ]
    }
   ],
   "source": [
    "# This Python 3 environment comes with many helpful analytics libraries installed\n",
    "# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python\n",
    "# For example, here's several helpful packages to load in \n",
    "\n",
    "import numpy as np # linear algebra\n",
    "import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)\n",
    "\n",
    "# Input data files are available in the \"../input/\" directory.\n",
    "# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory\n",
    "\n",
    "import os\n",
    "for dirname, _, filenames in os.walk('/kaggle/input'):\n",
    "    for filename in filenames:\n",
    "        print(os.path.join(dirname, filename))\n",
    "\n",
    "# Any results you write to the current directory are saved as output."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "ddb44fee",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2022-12-06T21:41:12.519936Z",
     "iopub.status.busy": "2022-12-06T21:41:12.519505Z",
     "iopub.status.idle": "2022-12-06T21:41:12.565293Z",
     "shell.execute_reply": "2022-12-06T21:41:12.564235Z"
    },
    "papermill": {
     "duration": 0.052044,
     "end_time": "2022-12-06T21:41:12.567904",
     "exception": false,
     "start_time": "2022-12-06T21:41:12.515860",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>PassengerId</th>\n",
       "      <th>Survived</th>\n",
       "      <th>Pclass</th>\n",
       "      <th>Name</th>\n",
       "      <th>Sex</th>\n",
       "      <th>Age</th>\n",
       "      <th>SibSp</th>\n",
       "      <th>Parch</th>\n",
       "      <th>Ticket</th>\n",
       "      <th>Fare</th>\n",
       "      <th>Cabin</th>\n",
       "      <th>Embarked</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>3</td>\n",
       "      <td>Braund, Mr. Owen Harris</td>\n",
       "      <td>male</td>\n",
       "      <td>22.0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>A/5 21171</td>\n",
       "      <td>7.2500</td>\n",
       "      <td>NaN</td>\n",
       "      <td>S</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>Cumings, Mrs. John Bradley (Florence Briggs Th...</td>\n",
       "      <td>female</td>\n",
       "      <td>38.0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>PC 17599</td>\n",
       "      <td>71.2833</td>\n",
       "      <td>C85</td>\n",
       "      <td>C</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>3</td>\n",
       "      <td>1</td>\n",
       "      <td>3</td>\n",
       "      <td>Heikkinen, Miss. Laina</td>\n",
       "      <td>female</td>\n",
       "      <td>26.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>STON/O2. 3101282</td>\n",
       "      <td>7.9250</td>\n",
       "      <td>NaN</td>\n",
       "      <td>S</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>4</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>Futrelle, Mrs. Jacques Heath (Lily May Peel)</td>\n",
       "      <td>female</td>\n",
       "      <td>35.0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>113803</td>\n",
       "      <td>53.1000</td>\n",
       "      <td>C123</td>\n",
       "      <td>S</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>5</td>\n",
       "      <td>0</td>\n",
       "      <td>3</td>\n",
       "      <td>Allen, Mr. William Henry</td>\n",
       "      <td>male</td>\n",
       "      <td>35.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>373450</td>\n",
       "      <td>8.0500</td>\n",
       "      <td>NaN</td>\n",
       "      <td>S</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   PassengerId  Survived  Pclass  \\\n",
       "0            1         0       3   \n",
       "1            2         1       1   \n",
       "2            3         1       3   \n",
       "3            4         1       1   \n",
       "4            5         0       3   \n",
       "\n",
       "                                                Name     Sex   Age  SibSp  \\\n",
       "0                            Braund, Mr. Owen Harris    male  22.0      1   \n",
       "1  Cumings, Mrs. John Bradley (Florence Briggs Th...  female  38.0      1   \n",
       "2                             Heikkinen, Miss. Laina  female  26.0      0   \n",
       "3       Futrelle, Mrs. Jacques Heath (Lily May Peel)  female  35.0      1   \n",
       "4                           Allen, Mr. William Henry    male  35.0      0   \n",
       "\n",
       "   Parch            Ticket     Fare Cabin Embarked  \n",
       "0      0         A/5 21171   7.2500   NaN        S  \n",
       "1      0          PC 17599  71.2833   C85        C  \n",
       "2      0  STON/O2. 3101282   7.9250   NaN        S  \n",
       "3      0            113803  53.1000  C123        S  \n",
       "4      0            373450   8.0500   NaN        S  "
      ]
     },
     "execution_count": 2,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "train_data = pd.read_csv(\"/kaggle/input/titanic/train.csv\")\n",
    "train_data.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "a6e53d82",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2022-12-06T21:41:12.575435Z",
     "iopub.status.busy": "2022-12-06T21:41:12.575089Z",
     "iopub.status.idle": "2022-12-06T21:41:12.596230Z",
     "shell.execute_reply": "2022-12-06T21:41:12.595175Z"
    },
    "papermill": {
     "duration": 0.027557,
     "end_time": "2022-12-06T21:41:12.598515",
     "exception": false,
     "start_time": "2022-12-06T21:41:12.570958",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>PassengerId</th>\n",
       "      <th>Pclass</th>\n",
       "      <th>Name</th>\n",
       "      <th>Sex</th>\n",
       "      <th>Age</th>\n",
       "      <th>SibSp</th>\n",
       "      <th>Parch</th>\n",
       "      <th>Ticket</th>\n",
       "      <th>Fare</th>\n",
       "      <th>Cabin</th>\n",
       "      <th>Embarked</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>892</td>\n",
       "      <td>3</td>\n",
       "      <td>Kelly, Mr. James</td>\n",
       "      <td>male</td>\n",
       "      <td>34.5</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>330911</td>\n",
       "      <td>7.8292</td>\n",
       "      <td>NaN</td>\n",
       "      <td>Q</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>893</td>\n",
       "      <td>3</td>\n",
       "      <td>Wilkes, Mrs. James (Ellen Needs)</td>\n",
       "      <td>female</td>\n",
       "      <td>47.0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>363272</td>\n",
       "      <td>7.0000</td>\n",
       "      <td>NaN</td>\n",
       "      <td>S</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>894</td>\n",
       "      <td>2</td>\n",
       "      <td>Myles, Mr. Thomas Francis</td>\n",
       "      <td>male</td>\n",
       "      <td>62.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>240276</td>\n",
       "      <td>9.6875</td>\n",
       "      <td>NaN</td>\n",
       "      <td>Q</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>895</td>\n",
       "      <td>3</td>\n",
       "      <td>Wirz, Mr. Albert</td>\n",
       "      <td>male</td>\n",
       "      <td>27.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>315154</td>\n",
       "      <td>8.6625</td>\n",
       "      <td>NaN</td>\n",
       "      <td>S</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>896</td>\n",
       "      <td>3</td>\n",
       "      <td>Hirvonen, Mrs. Alexander (Helga E Lindqvist)</td>\n",
       "      <td>female</td>\n",
       "      <td>22.0</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>3101298</td>\n",
       "      <td>12.2875</td>\n",
       "      <td>NaN</td>\n",
       "      <td>S</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   PassengerId  Pclass                                          Name     Sex  \\\n",
       "0          892       3                              Kelly, Mr. James    male   \n",
       "1          893       3              Wilkes, Mrs. James (Ellen Needs)  female   \n",
       "2          894       2                     Myles, Mr. Thomas Francis    male   \n",
       "3          895       3                              Wirz, Mr. Albert    male   \n",
       "4          896       3  Hirvonen, Mrs. Alexander (Helga E Lindqvist)  female   \n",
       "\n",
       "    Age  SibSp  Parch   Ticket     Fare Cabin Embarked  \n",
       "0  34.5      0      0   330911   7.8292   NaN        Q  \n",
       "1  47.0      1      0   363272   7.0000   NaN        S  \n",
       "2  62.0      0      0   240276   9.6875   NaN        Q  \n",
       "3  27.0      0      0   315154   8.6625   NaN        S  \n",
       "4  22.0      1      1  3101298  12.2875   NaN        S  "
      ]
     },
     "execution_count": 3,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "test_data = pd.read_csv(\"/kaggle/input/titanic/test.csv\")\n",
    "test_data.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "b78dccc8",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2022-12-06T21:41:12.606361Z",
     "iopub.status.busy": "2022-12-06T21:41:12.605960Z",
     "iopub.status.idle": "2022-12-06T21:41:12.619505Z",
     "shell.execute_reply": "2022-12-06T21:41:12.618273Z"
    },
    "papermill": {
     "duration": 0.019981,
     "end_time": "2022-12-06T21:41:12.621628",
     "exception": false,
     "start_time": "2022-12-06T21:41:12.601647",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "% of women who survived: 0.7420382165605095\n"
     ]
    }
   ],
   "source": [
    "women = train_data.loc[train_data.Sex == 'female'][\"Survived\"]\n",
    "rate_women = sum(women)/len(women)\n",
    "\n",
    "print(\"% of women who survived:\", rate_women)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "42e5cfd9",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2022-12-06T21:41:12.630274Z",
     "iopub.status.busy": "2022-12-06T21:41:12.629155Z",
     "iopub.status.idle": "2022-12-06T21:41:12.637043Z",
     "shell.execute_reply": "2022-12-06T21:41:12.635831Z"
    },
    "papermill": {
     "duration": 0.014344,
     "end_time": "2022-12-06T21:41:12.639194",
     "exception": false,
     "start_time": "2022-12-06T21:41:12.624850",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "% of men who survived: 0.18890814558058924\n"
     ]
    }
   ],
   "source": [
    "men = train_data.loc[train_data.Sex == 'male'][\"Survived\"]\n",
    "rate_men = sum(men)/len(men)\n",
    "\n",
    "print(\"% of men who survived:\", rate_men)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "2306a5b2",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2022-12-06T21:41:12.647779Z",
     "iopub.status.busy": "2022-12-06T21:41:12.647161Z",
     "iopub.status.idle": "2022-12-06T21:41:14.100922Z",
     "shell.execute_reply": "2022-12-06T21:41:14.099633Z"
    },
    "papermill": {
     "duration": 1.460919,
     "end_time": "2022-12-06T21:41:14.103476",
     "exception": false,
     "start_time": "2022-12-06T21:41:12.642557",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Your submission was successfully saved!\n"
     ]
    }
   ],
   "source": [
    "from sklearn.ensemble import RandomForestClassifier\n",
    "\n",
    "y = train_data[\"Survived\"]\n",
    "\n",
    "features = [\"Pclass\", \"Sex\", \"SibSp\", \"Parch\"]\n",
    "X = pd.get_dummies(train_data[features])\n",
    "X_test = pd.get_dummies(test_data[features])\n",
    "\n",
    "model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)\n",
    "model.fit(X, y)\n",
    "predictions = model.predict(X_test)\n",
    "\n",
    "output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})\n",
    "output.to_csv('submission.csv', index=False)\n",
    "print(\"Your submission was successfully saved!\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "68f3a84d",
   "metadata": {
    "papermill": {
     "duration": 0.003045,
     "end_time": "2022-12-06T21:41:14.110086",
     "exception": false,
     "start_time": "2022-12-06T21:41:14.107041",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.9.12"
  },
  "papermill": {
   "default_parameters": {},
   "duration": 10.515042,
   "end_time": "2022-12-06T21:41:14.835823",
   "environment_variables": {},
   "exception": null,
   "input_path": "__notebook__.ipynb",
   "output_path": "__notebook__.ipynb",
   "parameters": {},
   "start_time": "2022-12-06T21:41:04.320781",
   "version": "2.3.4"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
