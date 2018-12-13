#!/usr/bin/awk -f
BEGIN {
	print "Processing data file...";
	FS=";"
	test_count = 0;
	train_count = 0;
}

NR==1 { next }

NR%4 == 0 { out_file = "wiki.test.tmp"; test_count++; }
NR%4 != 0 { out_file = "wiki.train.tmp"; train_count++; }

{
## $1 is AGE: converted to float by dividing by 100
## $2 (OUTPUT)
## $3 is DOMAIN: skipped
## $4 (OUTPUT)
## $5 is YEARS OF EXP: converted to float by dividing by 50
## $6 is UNI: skipped

	printf("%.3f %.3f ", ($1/100), ($5/50)) > out_file;

## $7 is UNI_POSITION: 1=Professor 2=Associate 3=Assistant 4=Lecturer 5=Instructor 6=Adjunct
##	 converted to 1.0 0.8 0.6 0.4 0.2 and 0.0 respectively (0.0 if "?", which means different UNI)

	#if($7!="?")
	if($6=="1")
		printf("%.3f ", (6-$7)/5) >> out_file;
	else
		printf("0.000 ") >> out_file;

## $8 is OTHER_POS: skipped
## $9 is OTHER_STATUS: skipped
## $10 (OUTPUT)
## $11 to $54 are answers to surveys
	for (i = 11; i < NF; i++)
		if($i == "?")
			printf("0.000 ") >> out_file;
		else
			printf("%.3f ", $i/5) >> out_file;


## $2 is GENDER: 0 = Male and 1 = Female (OUTPUT)
## $4 is PhD: 0 = No and 1 = Yes (OUTPUT)
## $10 is USERWIKI: 0 = No and 1 = Registered (OUTPUT)
	if($10 == "?")
		ah = 0
	else
		ah = $10

	print $2, $4, ah >> out_file;
}

END {
	print test_count, 45, 3 > "wiki.test"
	print train_count, 45, 3 > "wiki.train"

	system("cat wiki.test.tmp >> wiki.test && rm wiki.test.tmp")
	system("cat wiki.train.tmp >> wiki.train && rm wiki.train.tmp")
}
