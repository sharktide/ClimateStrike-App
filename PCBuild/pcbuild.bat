dotnet publish STRIKE/StrikeAI.csproj -f net9.0-windows10.0.19041.0
tar -a -c -f "redist/ClimateStrike_AI-x64-redist.zip" "./STRIKE/bin/Release/net9.0-windows10.0.19041.0"
iscc PCBuild/Setup-amd64.iss
