@echo off
echo =====================================
echo Running Catboost.py
echo =====================================
python Catboost.py
if %errorlevel% neq 0 (
    echo "Catboost.py encountered an error."
    goto error
)

echo =====================================
echo Running RandomForest.py
echo =====================================
python RandomForest.py
if %errorlevel% neq 0 (
    echo "RandomForest.py encountered an error."
    goto error
)

echo =====================================
echo Running SVM.py
echo =====================================
python SVM.py
if %errorlevel% neq 0 (
    echo "SVM.py encountered an error."
    goto error
)

echo =====================================
echo Running TabPFN.py
echo =====================================
python TabPFN.py
if %errorlevel% neq 0 (
    echo "TabPFN.py encountered an error."
    goto error
)

echo =====================================
echo Running Xgboost.py
echo =====================================
python Xgboost.py
if %errorlevel% neq 0 (
    echo "Xgboost.py encountered an error."
    goto error
)

echo =====================================
echo All scripts ran successfully!
echo =====================================
goto end

:error
echo =====================================
echo One of the scripts failed to run.
echo Batch script will now exit.
echo =====================================

:end
pause
