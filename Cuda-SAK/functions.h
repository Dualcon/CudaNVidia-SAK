/** @file functions.h
 *  @brief The header file that contains the prototypes for functions.c
 * 
 *  @version 1
*/

#ifndef FUNCTIONS_H_INCLUDED
#define FUNCTIONS_H_INCLUDED
#define _XOPEN_SOURCE 700

char * device(void);
char * handleError(void);
void filename(char * text, char * fname);
char * callKernel(char * name);
char * genMaps(int dimension);
char * occupancy (void);
char * preprocessor (int arch);
char * reduction(void);
char * skeleton (char * kernel_name);
char * timing(void);
char * unified(void);
void students(const char * fname, char * stringToWrite, int lineToWrite);
char * about(void);
char * testAll(void);
char * buildString(char ** array, int length);

#endif
