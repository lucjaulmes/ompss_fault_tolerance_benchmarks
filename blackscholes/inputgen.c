//Copyright (c) 2009 Princeton University
//Written by Christian Bienia
//Generate input files for blackscholes benchmark

#include <stdio.h>
#include <stdlib.h>
#include <err.h>


//Precision to use
#define fptype double

typedef struct OptionData_
{
	fptype s;         // spot price
	fptype strike;    // strike price
	fptype r;         // risk-free interest rate
	fptype divq;      // dividend rate
	fptype v;         // volatility
	fptype t;         // time to maturity or option expiration in years (1yr = 1.0, 6mos = 0.5, 3mos = 0.25, ..., etc)
	char OptionType;  // Option type.  "P"=PUT, "C"=CALL
	fptype divs;      // dividend vals (not used in this test)
	fptype DGrefval;  // DerivaGem Reference Value
} OptionData;


OptionData data_init[] =
{
#include "optionData.txt"
};

//Total number of options in optionData.txt
#define MAX_OPTIONS (sizeof(data_init)/sizeof(*data_init))


int main(int argc, char **argv)
{
	if (argc != 3)
		errx(1, "Usage:\n\t%s <numOptions> <fileName>", argv[0]);

	int numOptions = atoi(argv[1]);
	if (numOptions < 1)
		errx(1, "ERROR: Number of options must at least be 1.");

	char *fileName = argv[2];
	FILE *file = fopen(fileName, "w");
	if (file == NULL)
		err(1, "ERROR: Unable to open file `%s'.", fileName);

	//write number of options
	if (fprintf(file, "%i\n", numOptions) < 0)
		err(1, "ERROR: Unable to write to file `%s'.", fileName);

	//write values for options
	for (int i = 0; i < numOptions; i++)
	{
        int j = i % MAX_OPTIONS;
		//NOTE: DG RefValues specified exceed double precision, output will deviate
		if (fprintf(file, "%.2f %.2f %.4f %.2f %.2f %.2f %c %.2f %.18f\n", data_init[j].s, data_init[j].strike, data_init[j].r, data_init[j].divq, data_init[j].v, data_init[j].t, data_init[j].OptionType, data_init[j].divs, data_init[j].DGrefval) < 0)
			err(1, "ERROR: Unable to write to file `%s'.", fileName);
	}

	if (fclose(file) != 0)
		err(1, "ERROR: Unable to close file `%s'.", fileName);

	return 0;
}
