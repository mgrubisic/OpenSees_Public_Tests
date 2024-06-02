@echo off
cd C:\Github\OpenSees_Public_Tests

echo Dodavanje izmjena u staging area...
git add .

echo Commitanje izmjena...
git commit -m "Update OpenSees_Public_Tests"

echo Slanje izmjena na GitHub...
git push origin main

echo Povlacenje najnovijih izmjena iz GitHub repozitorija...
git pull origin main

echo Gotovo!
pause
