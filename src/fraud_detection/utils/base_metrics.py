"""
Módulo auxiliar para calcular as métricas do negócio.

Funções
-------
- calculate_revenue: Calcula o faturamento da amostra a partir de um limiar.
- find_best_threshold: Encontra o melhor limiar a partir da amostra
- get_incoming_pressure_rate: Calcula a pressão de entrada de fraudes.
- get_approval_rate: Calcula a taxa de aprovação.
- get_decline_rate: Calcula a taxa de declínio de transações.
- get_precision: Calcula a precisão do modelo.
- get_recall: Calcula a revocação do modelo.
- get_false_positive_rate: Calcula a taxa de falsos positivos do modelo base.
- show_all_metrics: Exibe todas as métricas de avaliação do modelo base.
"""

import pandas as pd


class BaseMetrics:
    """
    Classe utilizada para calcular métricas de avaliação para modelo base
    de detecção de fraude.

    Possui funcionalidades para achar o melhor limiar de corte de avaliação,
    além de trazer métricas dos KPIs de interesse para a avaliação do modelo.

    Iremos utilizar tais métricas para comparar com o novo modelo criado.

    Args:
        data: DataFrame com as informações de compras realizadas
        score_column: Coluna com score de detecção de fraude do modelo base
        fraud_column: Nome da coluna com a classificação verdadeira de fraude
        value_column: Nome da coluna com o valor da transação
    """

    def __init__(
        self,
        data: pd.DataFrame,
        score_column: str,
        fraud_column: str,
        value_column: str,
    ):
        self.data = data
        self.score_column = score_column
        self.fraud_column = fraud_column
        self.value_column = value_column

        self.default_threshold = 75
        self.best_threshold = None

    def __check_best_threshold(self):
        """
        Função que verifica se o melhor limiar foi calculado previamente.

        Caso contrário utiliza o limiar encontrado como padrão no entendimento\
        do negócio.
        """
        if self.best_threshold is None:
            print(
                """Melhor limiar ainda não calculado.
                 Utilize o método find_best_threshold()"""
            )
            print("Utilizando limiar de corte padrão de 75")

            return self.default_threshold
        return self.best_threshold

    def calculate_revenue(self, threshold):
        """
        Função que calcula faturamento da amostra a partir de um limiar.

        O cálculo de faturamento se baseia nas premissas:
            - 10% do valor de compras aceitas é recebido.
            - 100% do valor de compras fraudulentar aceitas é perdido.

        Args:
            threshold (int): Limiar de score para aceitação.

        Returns:
            total_revenue (float): Faturamento total para o limiar fornecido.
        """

        try:
            accepted_sales = self.data.loc[
                self.data[self.score_column] < threshold
            ]

            total_normal_accepted = accepted_sales.loc[
                accepted_sales[self.fraud_column] == 0, self.value_column
            ].sum()

            total_fraud_accepted = accepted_sales.loc[
                accepted_sales[self.fraud_column] == 1, self.value_column
            ].sum()

            # Multiplica-se por 0.1 pois apenas 10% do valor é recebido.
            total_income = 0.1 * total_normal_accepted
            total_revenue = total_income - total_fraud_accepted
        except KeyError as e:
            raise KeyError(
                f"Colunas não encontradas no DataFrame fornecido. {e}"
            ) from e

        return total_income, total_fraud_accepted, total_revenue

    def find_best_threshold(self):
        """
        Função que encontra o melhor limiar de corte para o modelo base.

        O melhor limiar maximiza a receita gerada pelo modelo.
        """

        revenue_list = []
        for threshold in range(1, 100):
            revenue_list.append(
                self.calculate_revenue(threshold),
            )

        revenue_df = pd.DataFrame(
            revenue_list, columns=["income", "loss", "revenue"]
        )
        self.best_threshold = revenue_df["revenue"].idxmax()

        best_record = revenue_df.iloc[self.best_threshold]
        best_income = best_record["income"]
        minimum_loss = best_record["loss"]
        best_revenue = best_record["revenue"]

        print(
            "O limiar ótimo encontrado para a amostra é de: %s\n"
            "Ganhos por transações aprovadas: R$ %.2f\n"
            "Prejuízos com transações fraudulentas aprovadas: R$ %.2f\n"
            "Receita gerada com limiar ótimo: R$ %.2f",
            self.best_threshold,
            best_income,
            minimum_loss,
            best_revenue,
        )

        return self.best_threshold

    def get_incoming_pressure_rate(self):
        """
        Função que calcula a pressão de entrada de transações fraudulentas.

        Calculada como a razão entre o número de transações fraudulentas com\
        o número total de transações.

        Returns:
            incoming_pressure (float): Taxa de pressão de entrada de\
                                       transações.
        """

        incoming_pressure = (
            self.data[self.fraud_column].sum() / self.data.shape[0]
        ) * 100
        print(f"Taxa de pressão de entrada é de {incoming_pressure}%")

        return incoming_pressure

    def get_approval_rate(self):
        """
        Função que calcula a taxa de aprovação.

        Calculada como a razão entre o número de transações aprovadas e\
        o número total de transações.

        Returns:
            approval_rate (float): Taxa de aprovação.
        """
        best_threshold = self.__check_best_threshold()

        approved_count = self.data.loc[
            self.data[self.score_column] < best_threshold
        ].shape[0]
        approval_rate = (approved_count / self.data.shape[0]) * 100
        print(f"Taxa de aprovação total é de {approval_rate:.2f}%")

        return approval_rate

    def get_decline_rate(self):
        """
        Função que calcula a taxa de declínio de transações.

        Calculada como a razão entre o número de transações declinadas\
        e o número total de transações.
        
        Returns:
            decline_rate (float): Taxa de declínio.
        """

        best_threshold = self.__check_best_threshold()

        declined_count = self.data.loc[
            self.data[self.score_column] > best_threshold
        ].shape[0]
        decline_rate = (declined_count / self.data.shape[0]) * 100
        print(f"Taxa de declínio total é de {decline_rate:.2f}%")

        return decline_rate

    def get_precision(self):
        """
        Função que calcula a precisão do modelo.

        VP: fraudes identificadas corretamente
        FP: transações legítimas identificadas como fraudes

        Precisão = VP / (VP + FP)

        Returns:
            precision (float): Taxa de precisão.
        """

        best_threshold = self.__check_best_threshold()

        declined_sales = self.data.loc[
            self.data[self.score_column] > best_threshold
        ]
        vp = declined_sales.loc[(self.data[self.fraud_column] == 1)].shape[0]
        fp = declined_sales.loc[(self.data[self.fraud_column] == 0)].shape[0]

        precision = vp / (vp + fp) * 100
        print(f"A precisão do modelo é de {precision:.2f}%")

        return precision

    def get_recall(self):
        """
        Função que calcula a revocação do modelo.

        VP: fraudes identificadas corretamente
        FN: transações fraudulentas identificadas como legítimas

        Recall = VP / (VP + FN)

        Returns:
            recall (float): Taxa de revocação.
        """

        best_threshold = self.__check_best_threshold()

        declined_sales = self.data.loc[
            self.data[self.score_column] > best_threshold
        ]
        vp = declined_sales.loc[(self.data[self.fraud_column] == 1)].shape[0]

        accepted_sales = self.data.loc[
            self.data[self.score_column] < best_threshold
        ]
        fn = accepted_sales.loc[(self.data[self.fraud_column] == 1)].shape[0]

        precision = vp / (vp + fn) * 100
        print(f"A revocação do modelo é de {precision:.2f}%")

        return precision

    def get_false_positive_rate(self):
        """
        Função que calcula a taxa de falsos positivos do modelo base.

        FP: transações legítimas identificadas como fraudes
        VN: transações legítimas identificadas corretamente

        FPR = FP / (FP + VN)

        Returns:
            false_positive_rate (float): Taxa de falsos positivos.
        """
        best_threshold = self.__check_best_threshold()

        declined_sales = self.data.loc[
            self.data[self.score_column] > best_threshold
        ]
        fp = declined_sales.loc[(self.data[self.fraud_column] == 0)].shape[0]

        accepted_sales = self.data.loc[
            self.data[self.score_column] < best_threshold
        ]
        vn = accepted_sales.loc[(self.data[self.fraud_column] == 0)].shape[0]

        false_positive_rate = fp / (fp + vn) * 100
        print(f"A taxa de falsos positivos é de {false_positive_rate:.2f}%")

        return false_positive_rate

    def show_all_metrics(self):
        """
        Função que exibe todas as métricas de avaliação do modelo base.
        """
        if self.best_threshold is None:
            self.find_best_threshold()

        self.get_incoming_pressure_rate()
        self.get_approval_rate()
        self.get_decline_rate()
        self.get_precision()
        self.get_recall()
        self.get_false_positive_rate()
