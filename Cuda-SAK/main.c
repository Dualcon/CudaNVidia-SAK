/**
* @file main.c
* @brief File with the main function where it's decided what function to call from cuda-sak options.
* 
* @date 13/12/2014
* @author Rafael Costa, 2100073@my.ipleiria.pt
* @author Nelson Nunes, 2141481@my.ipleiria.pt
* @author Daniel Vieira, 2141472@my.ipleiria.pt
* @author David Matos, 2100395@my.ipleiria.pt
*/

#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
#include <signal.h>
#include <errno.h>
#include <string.h>
#include <sys/types.h>
#include <sys/ipc.h>
#include <sys/sem.h>
#include <sys/shm.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <assert.h>

#include "debug.h"
#include "memory.h"
#include "cuda-sakgengetopt.h"
#include "functions.h"

int main(int argc, char *argv[]) {

	struct gengetopt_args_info ArgsInfo;
	char  *output;

	if ( cmdline_parser(argc, argv, &ArgsInfo) ) {

		fprintf(stderr,"Error: cmdline_parser execution\n");
		exit(EXIT_FAILURE);
	}

	if (ArgsInfo.device_given ) {

		output = device();
	}

	if (ArgsInfo.handleerror_given ) {

		output = handleError();
	}

	if (ArgsInfo.callkernel_given) {

		output = callKernel(ArgsInfo.callkernel_arg);
	}

	if (ArgsInfo.genmaps_given) {

		output = genMaps(ArgsInfo.genmaps_arg);
	}

	if (ArgsInfo.occupancy_given) {

		output = occupancy();
	}

	if (ArgsInfo.preprocessor_given) {

		if ( (ArgsInfo.preprocessor_arg % 10) == 0 && ArgsInfo.preprocessor_arg > 1 )
			output = preprocessor(ArgsInfo.preprocessor_arg);
		else {
			printf("%s", "Invalid architecture number! It must be a multiple of 10 and greater than 1.\n");
			cmdline_parser_free(&ArgsInfo);
			exit(EXIT_FAILURE);
		}
	}

	if (ArgsInfo.reduction_given) {

		output = reduction();
	}

	if (ArgsInfo.skeleton_given) {

		output = skeleton(ArgsInfo.skeleton_arg);
	}

	if (ArgsInfo.timing_given) {

		output = timing();
	}

	if (ArgsInfo.unified_given) {

		output = unified();
	}

	if (ArgsInfo.about_given) {

		output = about();
	}

	if (ArgsInfo.testall_given) {

		output = testAll();
	}

	if (ArgsInfo.filename_given) {

		filename(output, ArgsInfo.filename_arg);
		cmdline_parser_free(&ArgsInfo);

	} else if (ArgsInfo.students_given) {

		char * result = strtok(ArgsInfo.students_arg, " ");
		char * params[2];

		int i = 0;

		while (result != NULL) {

			params[i] = result;
			i++;
			result = strtok(NULL, " ");
		}

		int lineToWrite = atoi(params[1]);

		if (lineToWrite == 0) {

			fprintf(stderr,"Error: Second parameter(line to write) can't be converted to integer.\n");
			free(output);
			cmdline_parser_free(&ArgsInfo);
			exit(EXIT_FAILURE);
		}

		students(params[0], output, lineToWrite);
		cmdline_parser_free(&ArgsInfo);

	} else {

		printf("%s", output);
		FREE(output);
		cmdline_parser_free(&ArgsInfo);

		if (!ArgsInfo.about_given) {

			printf("----------------------------------------------------\n");
			printf("CUDA - Swiss Army Knife. (--help to show options)\n");
		}
	}

	return 0;
}

