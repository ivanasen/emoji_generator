build-archive:
	mkdir -p bin
	zip -r ./bin/emoji_generator.zip ./src/* ./notebooks/final.ipynb ./README.md
