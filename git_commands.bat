@echo off
cd /d %~dp0
git add .
git commit -m "Initial commit with optimized repository"
git branch -M main
git remote add origin https://github.com/alyyemecoder/arascan.git
git push -u origin main
