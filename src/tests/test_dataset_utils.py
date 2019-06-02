from ..utils import dataset_utils


class TestDatasetUtils(object):
    """Collects all unit tests for `utils.model_utils`.
    """
    def test_load_wikihop(self, dataset):
        """Asserts that `data_utils.load_wikihop()` returns the expected value when `masked == True`.
        """
        expected = {'train': [
            {
                "id": "WH_train_0",
                "query": "participant_of juan rossell",
                "answer": "1996 summer olympics",
                "candidates": [
                    "1996 summer olympics",
                    "olympic games",
                    "sport"
                ],
                "supports": [
                    "Juan Miguel Rossell Milanes ( born December 28 , 1969 in Jiguani , Granma ) is a beach volleyball player from Cuba , who won the gold medal in the men 's beach team competition at the 2003 Pan American Games in Santo Domingo , Dominican Republic , partnering Francisco Alvarez . He represented his native country at the 1996 and the 2004 Summer Olympics ."
                ]
            },
            {
                "id": "WH_train_1",
                "query": "languages_spoken_or_written john osteen",
                "answer": "english",
                "candidates": [
                    "english",
                    "greek",
                    "koine greek",
                    "nahuatl",
                    "spanish"
                ],
                "supports": [
                    "Lakewood Church is a nondenominational charismatic Christian megachurch located in Houston, Texas. It is the largest congregation in the United States, averaging about 52,000 attendees per week. The 16,800-seat Lakewood Church Central Campus, home to four English-language services and two Spanish-language services per week, is located at the former Compaq Center. Joel Osteen is the senior pastor of Lakewood Church with his wife, Victoria, who serves as co-pastor. Lakewood Church is a part of the Word of Faith movement.",
                    "Mexico (, modern Nahuatl ), officially the United Mexican States, is a federal republic in the southern half of North America. It is bordered to the north by the United States; to the south and west by the Pacific Ocean; to the southeast by Guatemala, Belize, and the Caribbean Sea; and to the east by the Gulf of Mexico. Covering almost two million square kilometers (over 760,000\u00a0sq\u00a0mi), Mexico is the sixth largest country in the Americas by total area and the 13th largest independent nation in the world. With an estimated population of over 120 million, it is the eleventh most populous country and the most populous Spanish-speaking country in the world while being the second most populous country in Latin America. Mexico is a federation comprising 31 states and a federal district that is also its capital and most populous city. Other metropolises include Guadalajara, Monterrey, Puebla, Toluca, Tijuana and Le\u00f3n."
                ]
            }
        ]}

        actual = dataset

        assert expected == actual

    def test_load_wikihop_masked(self, masked_dataset):
        """Asserts that `data_utils.load_wikihop()` returns the expected value when `masked == False`.
        """
        expected = {'train.masked': [
            {
                "id": "WH_train_0",
                "query": "participant_of juan rossell",
                "answer": "___MASK3___",
                "candidates": [
                    "___MASK3___",
                    "___MASK63___",
                    "___MASK83___"
                ],
                "supports": [
                    "Juan Miguel Rossell Milanes ( born December 28 , 1969 in Jiguani , Granma ) is a beach volleyball player from Cuba , who won the gold medal in the men ' s beach team competition at the 2003 Pan American Games in Santo Domingo , Dominican Republic , partnering Francisco Alvarez . He represented his native country at the 1996 and the 2004 Summer Olympics ."
                ]
            },
            {
                "id": "WH_train_1",
                "query": "languages_spoken_or_written john osteen",
                "answer": "___MASK46___",
                "candidates": [
                    "___MASK15___",
                    "___MASK25___",
                    "___MASK46___",
                    "___MASK67___",
                    "___MASK85___"
                ],
                "supports": [
                    "Lakewood Church is a nondenominational charismatic Christian megachurch located in Houston , Texas . It is the largest congregation in the United States , averaging about 52 , 000 attendees per week . The 16 , 800 - seat Lakewood Church Central Campus , home to four ___MASK46___ - language services and two Spanish - language services per week , is located at the former Compaq Center . Joel Osteen is the senior pastor of Lakewood Church with his wife , Victoria , who serves as co - pastor . Lakewood Church is a part of the Word of Faith movement .",
                    "Mexico (, modern ___MASK67___ ), officially the United Mexican States , is a federal republic in the southern half of North America . It is bordered to the north by the United States ; to the south and west by the Pacific Ocean ; to the southeast by Guatemala , Belize , and the Caribbean Sea ; and to the east by the Gulf of Mexico . Covering almost two million square kilometers ( over 760 , 000 sq mi ), Mexico is the sixth largest country in the Americas by total area and the 13th largest independent nation in the world . With an estimated population of over 120 million , it is the eleventh most populous country and the most populous ___MASK25___ - speaking country in the world while being the second most populous country in Latin America . Mexico is a federation comprising 31 states and a federal district that is also its capital and most populous city . Other metropolises include Guadalajara , Monterrey , Puebla , Toluca , Tijuana and Le\u00f3n ."
                ]
            }
        ]}

        actual = masked_dataset

        assert expected == actual
