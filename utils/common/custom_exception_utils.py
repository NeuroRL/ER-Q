class InvalidJSONConfigException(Exception):
	def __init__(self, message):
		super().__init__(f"jsonの設定値に異常があります。hint: {message}")