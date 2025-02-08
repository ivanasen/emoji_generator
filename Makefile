build-archive:
	mkdir -p bin
	zip -r ./bin/emoji_generator.zip ./src/* ./notebooks/final.ipynb ./link_to_source_code.txt
