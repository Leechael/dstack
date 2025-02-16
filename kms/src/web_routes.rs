use crate::{main_service::KmsState, main_service::RpcHandler};
use ra_rpc::rocket_helper::{handle_prpc, QuoteVerifier};
use rocket::{
    data::{Data, Limits},
    get,
    http::ContentType,
    mtls::Certificate,
    post,
    response::status::Custom,
    routes, Route, State,
};

#[get("/")]
async fn index() -> String {
    "KMS Server is running!\n".to_string()
}

#[post("/prpc/<method>?<json>", data = "<data>")]
async fn prpc_post(
    state: &State<KmsState>,
    quote_verifier: Option<&State<QuoteVerifier>>,
    cert: Option<Certificate<'_>>,
    method: &str,
    data: Data<'_>,
    limits: &Limits,
    content_type: Option<&ContentType>,
    json: bool,
) -> Custom<Vec<u8>> {
    handle_prpc::<_, RpcHandler>(
        &*state,
        cert,
        quote_verifier.map(|v| &**v),
        method,
        Some(data),
        limits,
        content_type,
        json,
    )
    .await
}

#[get("/prpc/<method>")]
async fn prpc_get(
    state: &State<KmsState>,
    quote_verifier: Option<&State<QuoteVerifier>>,
    cert: Option<Certificate<'_>>,
    method: &str,
    limits: &Limits,
    content_type: Option<&ContentType>,
) -> Custom<Vec<u8>> {
    handle_prpc::<_, RpcHandler>(
        &*state,
        cert,
        quote_verifier.map(|v| &**v),
        method,
        None,
        limits,
        content_type,
        true,
    )
    .await
}

pub fn routes() -> Vec<Route> {
    routes![index, prpc_post, prpc_get]
}
