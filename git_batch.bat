@echo off
cd C:\Github\OpenSees_Public_Tests

set /p commitMessage="Update OpenSees_Public_Tests"

echo Dodavanje izmjena u staging area...
git add .

echo Commitanje izmjena...
git commit -m "%commitMessage%"

echo Slanje izmjena na GitHub...
git push origin main

echo Povlačenje najnovijih izmjena iz GitHub repozitorija...
git pull origin main

echo Gotovo!
pause
