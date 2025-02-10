This is a description of the project. 

## Pre-requisite operations

Because we're working in VSCode, Python packages may not be installed. Rather than installing them on local machine, we can make a new virtual environment for this project.

`python -m venv myenv`

Activate it: 

`myenv\Scripts\activate`

If there are security issues, check PowerShell's security settings and bypass if necessary:

`Get-ExecutionPolicy`
`Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`

Then install required packages inside the virual environment:

`pip install pandas numpy matplotlib seaborn scikit-learn torch matplotlib`
